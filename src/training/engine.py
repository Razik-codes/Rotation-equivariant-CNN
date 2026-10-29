"""Validation-only checkpoint selection, no test-set access."""
import os
import copy
import fcntl
import hashlib
import time
from pathlib import Path
import torch
import yaml
from src.common import (accuracy, build_model, environment, seed_everything,
                        validate_config, write_csv, write_json)
from src.datasets import training_data
from src.training.recovery import (TrainingInterrupted, atomic_save, rng_state, restore_rng)


def run_path(cfg):
    return (Path(cfg['output_dir']) / cfg['dataset'] / cfg['model'] /
            cfg['training_regime'] / str(cfg['seed']))


def train(cfg, resume=False, stop_requested=None):
    validate_config(cfg)
    if torch.device(cfg['device']).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Check nvidia-smi and install a CUDA-enabled PyTorch build; CPU fallback is disabled.')
    out = run_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode in place: unlinking lock files can permit concurrent writers.
    with (out.parent / f'.{out.name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Run is already active: {out}') from exc
        return _train(cfg, resume, stop_requested or (lambda: False))


def _train(cfg, resume, stop_requested):
    validate_config(cfg)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_everything(cfg['seed'], cfg['deterministic'])
    torch.set_num_threads(cfg['num_threads'])
    device = torch.device(cfg['device'])
    out = run_path(cfg)
    exists = out.exists()
    if exists and not resume:
        raise FileExistsError(f'Refusing to overwrite run: {out}. Use --resume or choose another output_dir.')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for root in ('src', 'scripts') for p in sorted(Path(root).rglob('*.py'))}
    saved = None
    if exists:
        if not (out / 'last.pt').exists():
            raise ValueError(f'No recovery checkpoint in {out}; existing evidence will not be overwritten.')
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if saved['config'] != cfg:
            raise ValueError('Resume configuration differs from the saved protocol')
        if saved['metadata']['source_hashes'] != source_hashes:
            raise ValueError('Source changed since checkpoint; restore the original source before resuming')
        current = environment()
        if any(saved['metadata']['environment'][key] != current[key] for key in ('versions', 'gpu')):
            raise ValueError('Software versions or GPU changed; exact continuation is not supported')
    train_loader, val_loader, manifest = training_data(cfg)
    model = build_model(cfg).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                             weight_decay=cfg['weight_decay'])
    if saved and saved['split_hash'] != manifest['hash']:
        raise ValueError('Training split differs from the recovery checkpoint')
    if saved and saved['metadata']['status'] == 'complete' and (out / 'best.pt').exists():
        # last.pt is authoritative if the process died before updating sidecars.
        write_json(out / 'metadata.json', saved['metadata'])
        write_csv(out / 'training_history.csv', saved['history'])
        print(f'Already complete: {out}', flush=True)
        return out / 'best.pt'
    out.mkdir(parents=True, exist_ok=True)
    if not saved:
        (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(out / 'split.json', manifest)
    metadata = dict(environment=environment(), source_hashes=source_hashes, dataset=cfg['dataset'],
                    split_hash=manifest['hash'], train_examples=len(train_loader.dataset),
                    validation_examples=len(val_loader.dataset), device=str(device),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    augmentation=cfg['augmentation'] if cfg['training_regime'] == 'rotation' else None,
                    status='running', checkpoint_rule='highest clean validation accuracy; earliest tie',
                    batch_size=cfg['batch_size'], precision='float32',
                    track=cfg.get('track', 'A-matched-steps'),
                    class_reports=manifest.get('class_reports'))
    seen_ids = set()
    step = epoch = exposures = 0
    best = -1.0
    history = []
    start = time.perf_counter()
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    train_seconds = 0.0
    interval_loss = interval_count = 0
    batch_index = 0
    epoch_generator = train_loader.generator.get_state()
    best_checkpoint = None
    previous_seconds = 0.0
    if saved:
        model.train()
        model.load_state_dict(saved['state_dict'], strict=True)
        optim.load_state_dict(saved['optimizer'])
        metadata = saved['metadata']
        step, epoch, exposures = saved['step'], saved['epoch'], saved['exposures']
        best, history = saved['best'], saved['history']
        seen_ids = set(saved['seen_ids'])
        train_seconds = saved['train_seconds']
        interval_loss, interval_count = saved['interval_loss'], saved['interval_count']
        batch_index, epoch_generator = saved['batch_index'], saved['epoch_generator']
        val_loader.generator.set_state(saved['val_generator'])
        previous_seconds = saved['elapsed_seconds']
        best_checkpoint = saved['best_checkpoint']
        if best_checkpoint is not None:
            atomic_save(best_checkpoint, out / 'best.pt')
        if history:
            write_csv(out / 'training_history.csv', history)
        print(f"Resuming {out} at step {step}/{cfg['max_steps']}", flush=True)
    metadata['status'] = 'running'
    write_json(out / 'metadata.json', metadata)

    def save_recovery(status='running'):
        model.train()  # exclude ESCNN's evaluation-only filter caches
        elapsed = previous_seconds + time.perf_counter() - start
        metadata.update(status=status, optimizer_steps=step, training_exposures=exposures,
                        unique_labeled_examples=len(seen_ids), train_seconds=train_seconds,
                        total_seconds=elapsed,
                        peak_gpu_memory_bytes=max(metadata.get('peak_gpu_memory_bytes') or 0,
                            torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None)
        atomic_save(dict(format_version=1, config=cfg, split_hash=manifest['hash'],
                         state_dict=model.state_dict(), optimizer=optim.state_dict(),
                         metadata=metadata, step=step, epoch=epoch, exposures=exposures,
                         best=best, best_checkpoint=best_checkpoint, history=history,
                         seen_ids=sorted(seen_ids), train_seconds=train_seconds,
                         elapsed_seconds=elapsed, interval_loss=interval_loss,
                         interval_count=interval_count, batch_index=batch_index,
                         epoch_generator=epoch_generator,
                         val_generator=val_loader.generator.get_state(), rng=rng_state()), out / 'last.pt')
        write_json(out / 'metadata.json', metadata)

    # A checkpoint also exists if interrupted before the first optimizer step.
    if not saved:
        save_recovery()
    while step < cfg['max_steps']:
        train_loader.dataset.epoch = epoch
        train_loader.generator.set_state(epoch_generator)
        batches = iter(train_loader)
        for _ in range(batch_index):
            next(batches)  # reconstruct this epoch's permutation and position
        if saved:
            restore_rng(saved['rng'])
            saved = None
        model.train()
        if stop_requested():
            save_recovery('interrupted')
            raise TrainingInterrupted(f'Saved {out / "last.pt"} at step {step}')
        for x, y, original_ids in batches:
            seen_ids.update(original_ids.tolist())
            sync()
            tick = time.perf_counter()
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optim.step()
            sync()
            train_seconds += time.perf_counter() - tick
            step += 1
            batch_index += 1
            exposures += len(y)
            interval_loss += float(loss.detach()) * len(y)
            interval_count += len(y)
            if step % cfg['validate_every'] == 0 or step == cfg['max_steps']:
                score = accuracy(model, val_loader, device)
                history.append(dict(step=step, epoch=epoch + 1, exposures=exposures,
                                    training_loss=interval_loss / interval_count,
                                    validation_accuracy=score, train_seconds=train_seconds))
                interval_loss = interval_count = 0
                if score > best:
                    best = score
                    # ESCNN eval creates cached filters; save the train-mode state only.
                    model.train()
                    payload = dict(state_dict=model.state_dict(), optimizer=optim.state_dict(),
                                   config=cfg, split_hash=manifest['hash'], step=step,
                                   validation_accuracy=score)
                    best_checkpoint = copy.deepcopy(payload)
                    atomic_save(best_checkpoint, out / 'best.pt')
                model.train()
                write_csv(out / 'training_history.csv', history)
                print(f"{cfg['model']}/{cfg['training_regime']}/{cfg['seed']} step={step} val={score:.4f}", flush=True)
            stopping = stop_requested()
            if (stopping or step % cfg.get('checkpoint_every', cfg['validate_every']) == 0
                    or step % cfg['validate_every'] == 0 or step == cfg['max_steps']):
                save_recovery('interrupted' if stopping else 'running')
            if stopping:
                raise TrainingInterrupted(f'Saved {out / "last.pt"} at step {step}')
            if step >= cfg['max_steps']:
                break
        if step < cfg['max_steps']:
            epoch += 1
            batch_index = 0
            epoch_generator = train_loader.generator.get_state()
    metadata.update(status='complete', optimizer_steps=step, training_exposures=exposures,
                    unique_labeled_examples=len(seen_ids), train_seconds=train_seconds,
                    total_seconds=previous_seconds + time.perf_counter() - start,
                    timing_scope='batch transfer + forward + backward + optimizer; excludes loading/validation',
                    gpu_training_seconds=train_seconds if device.type == 'cuda' else None)
    save_recovery('complete')
    return out / 'best.pt'
