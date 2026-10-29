"""Additional research figures, exclusively from saved measurement tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def additional_figures(rows, metadata, analysis, save):
    notes=[]
    selected=[r for r in rows if r['metric']=='mean_accuracy']
    fig,ax=plt.subplots(figsize=(9,4))
    for i,r in enumerate(selected):
        mean=float(r['mean'])
        if r['ci_low']:
            ax.errorbar(i,mean,yerr=[[mean-float(r['ci_low'])],[float(r['ci_high'])-mean]],fmt='o',capsize=4)
        else:
            ax.plot(i,mean,'o')
    ax.set_xticks(range(len(selected)),[f"{r['model']} / {r['training_regime']} (n={r['n']})" for r in selected],rotation=20,ha='right')
    ax.set_ylabel('Primary mean accuracy\n95% seed t CI when available')
    if any(not r['ci_low'] for r in selected):
        ax.set_title('Seed confidence intervals unavailable for n=1',fontsize=9)
        notes.append('Seed confidence intervals unavailable for groups with fewer than two seeds; points are means only.')
    save(fig,'mean_accuracy_confidence_intervals')
    augment=analysis/'augmentation_effects.csv'
    if augment.exists():
        selected=[r for r in read_csv(augment) if r['metric']=='mean_accuracy']
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(range(len(selected)),[float(r['mean']) for r in selected])
        for i,r in enumerate(selected):
            if r['ci_low']:
                ax.errorbar(i,float(r['mean']),yerr=[[float(r['mean'])-float(r['ci_low'])],[float(r['ci_high'])-float(r['mean'])]],color='black',capsize=4)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set_xticks(range(len(selected)),[r['contrast'] for r in selected])
        ax.set_ylabel('Rotation augmentation − original; mean accuracy')
        save(fig,'augmentation_improvements')
    else:
        notes.append('Augmentation improvement figure unavailable: no matched regime measurements.')
    fig,ax=plt.subplots(figsize=(7,4))
    for model in sorted({r['model'] for r in rows}):
        selected=[next((r for r in rows if r['model']==model and r['training_regime']==regime and r['metric']=='mean_accuracy'),None) for regime in ('original','rotation')]
        if all(selected):
            ax.plot([0,1],[float(r['mean']) for r in selected],'o-',label=model)
    ax.set_xticks([0,1],['Original training','C4 augmentation'])
    ax.set_ylabel('Four-angle mean accuracy')
    ax.legend()
    save(fig,'architecture_augmentation_interaction')
    cost_path=analysis/'training_costs.csv'
    if cost_path.exists():
        costs=read_csv(cost_path)
        groups=sorted({(r['model'],r['training_regime']) for r in costs})
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,metric,label in zip(axes,['parameters','train_seconds','inference_seconds'],['Trainable parameters','Training batch compute (seconds)','Test inference total (seconds)']):
            available=False
            for i,group in enumerate(groups):
                values=[float(r[metric]) for r in costs if (r['model'],r['training_regime'])==group and r.get(metric)]
                if values:
                    ax.bar(i,np.mean(values)); available=True
            if not available:
                ax.text(.5,.5,'Unavailable',transform=ax.transAxes,ha='center')
                notes.append(label+' unavailable in saved records.')
            ax.set_xticks(range(len(groups)),[f'{m}/{r}' for m,r in groups],rotation=60,ha='right',fontsize=7)
            ax.set_ylabel(label)
        save(fig,'parameters_and_measured_cost')
    fig,ax=plt.subplots(figsize=(8,4.5))
    count=0
    for folder in metadata['sources']:
        path=Path(folder)/'equivariance.csv'
        if not path.exists():
            continue
        diagnostics=[r for r in read_csv(path) if not r['layer'].startswith('classifier')]
        if not diagnostics:
            continue
        layer=diagnostics[-1]['layer']
        selected=sorted([r for r in diagnostics if r['layer']==layer],key=lambda r:float(r['angle']))
        ax.plot([float(r['angle']) for r in selected],[float(r['full_error']) for r in selected],'o-',label='/'.join(Path(folder).parts[-3:]))
        count+=1
    if count:
        ax.set(xlabel='Group rotation (degrees)',ylabel='Final feature layer relative L2 error')
        ax.set_yscale('symlog',linthresh=1e-7)
        ax.set_title('C8 non-right angles involve interpolation',fontsize=9)
        ax.legend(fontsize=7)
        save(fig,'equivariance_by_angle')
    else:
        plt.close(fig)
        notes.append('Equivariance figure unavailable: no saved feature diagnostics. Run the diagnostic CLI on validation data.')
    return notes
