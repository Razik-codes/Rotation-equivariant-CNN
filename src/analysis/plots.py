"""Generate figures exclusively from saved runs and statistical summaries."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(analysis, output):
    analysis, output = Path(analysis),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if not (analysis/'seed_summary.csv').exists():
        message='Figures unavailable: no saved seed_summary.csv measurements.'
        (output/'availability.txt').write_text(message+'\n')
        print(message)
        return
    with (analysis/'seed_summary.csv').open() as fh:
        rows=list(csv.DictReader(fh))
    groups=sorted({(r['model'],r['training_regime']) for r in rows})
    metadata=json.loads((analysis/'analysis_metadata.json').read_text())
    title='Saved experiment results'
    if any(any(label in p for label in ('validation', 'smoke', 'pilot')) for p in metadata['sources']):
        title='Pipeline validation — not research performance'
    def save(fig,name):
        fig.suptitle(title,fontsize=10)
        fig.tight_layout()
        fig.savefig(output/f'{name}.png',dpi=160,bbox_inches='tight')
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    for metric,ylabel in [('accuracy','Accuracy'),('macro_f1','Macro-F1'),('degradation_from_zero','Accuracy loss from 0°'),('output_consistency_l1','Mean probability L1 distance')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model,regime in groups:
            selected=sorted([r for r in rows if (r['model'],r['training_regime'])==(model,regime)
                             and r['metric'].startswith(metric+'@')],key=lambda r:float(r['metric'].split('@')[1]))
            angles=np.array([float(r['metric'].split('@')[1]) for r in selected])
            mean=np.array([float(r['mean']) for r in selected])
            ax.plot(angles,mean,'o-',label=f'{model} / {regime}')
            if all(r['sd'] for r in selected):
                sd=np.array([float(r['sd']) for r in selected])
                ax.fill_between(angles,mean-sd,mean+sd,alpha=.12)
        ax.set(xlabel='Test rotation (degrees)',ylabel=ylabel)
        ax.legend(fontsize=8,ncol=2)
        ax.grid(alpha=.2)
        save(fig,metric+'_by_angle')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for j,metric in enumerate(['clean_accuracy','rotated_accuracy']):
        selected=[next(r for r in rows if (r['model'],r['training_regime'])==g and r['metric']==metric) for g in groups]
        means=[float(r['mean']) for r in selected]
        sd=[float(r['sd']) if r['sd'] else 0 for r in selected]
        ax.bar(np.arange(len(groups))+(j-.5)*.36,means,.36,yerr=sd,label=metric.replace('_',' '),capsize=3)
    ax.set_xticks(np.arange(len(groups)),[f'{m}\n{r}' for m,r in groups],fontsize=8)
    ax.set_ylabel('Accuracy (mean ± seed SD; SD unavailable for n=1)')
    ax.legend()
    save(fig,'clean_vs_rotated')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for folder in metadata['sources']:
        path=Path(folder)
        with (path/'training_history.csv').open() as fh:
            history=list(csv.DictReader(fh))
        label='/'.join(path.parts[-3:])
        steps=[int(r['step']) for r in history]
        axes[0].plot(steps,[float(r['training_loss']) for r in history],label=label,alpha=.7)
        axes[1].plot(steps,[float(r['validation_accuracy']) for r in history],alpha=.7)
    axes[0].set(xlabel='Optimizer steps',ylabel='Training loss')
    axes[1].set(xlabel='Optimizer steps',ylabel='Clean validation accuracy')
    axes[0].legend(fontsize=5,ncol=2)
    save(fig,'training_curves')
    paired=analysis/'paired_seed_differences.csv'
    if paired.exists():
        with paired.open() as fh:
            selected=[r for r in csv.DictReader(fh) if r['metric']=='mean_accuracy' and r['ci_low']]
        if selected:
            fig,ax=plt.subplots(figsize=(9,max(3,len(selected)*.4)))
            means=np.array([float(r['mean']) for r in selected])
            errors=np.array([[m-float(r['ci_low']) for m,r in zip(means,selected)],
                             [float(r['ci_high'])-m for m,r in zip(means,selected)]])
            ax.errorbar(means,np.arange(len(selected)),xerr=errors,fmt='o',capsize=3)
            ax.axvline(0,color='gray',linestyle='--')
            ax.set_yticks(np.arange(len(selected)),[r['comparison'] for r in selected],fontsize=7)
            ax.set_xlabel('Paired mean-angle accuracy difference, 95% seed t interval')
            save(fig,'paired_confidence_intervals')
    from src.analysis.research_plots import additional_figures
    notes=additional_figures(rows,metadata,analysis,save)
    (output/'availability.txt').write_text('\n'.join(notes)+'\n')
    (output/'README.txt').write_text('Source: '+str(analysis)+'\nAngle bands and bars: sample SD across training seeds, not confidence intervals.\nPaired intervals: unadjusted 95% Student-t intervals across paired seeds.\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    plot_results(args.analysis,args.output)


if __name__ == '__main__':
    main()
