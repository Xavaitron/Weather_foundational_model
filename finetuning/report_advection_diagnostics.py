"""Render the completed diagnostic evidence; no forecast or training changes."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/advection-diagnostics-20260928'))
    p.add_argument('--training-root',type=Path,default=Path('runs/full-1deg-stage1-20260927'))
    a=p.parse_args();root=a.root
    status=json.loads((root/'run.json').read_text())
    assert status['status']=='complete' and len(status['completed_dates'])==12
    changes=json.loads((root/'parameter_changes.json').read_text())
    d=json.loads((root/'displacement.json').read_text())
    initial=[r for r in d if r['endpoint']=='initial'];trained=[r for r in d if r['endpoint']=='trained']
    avg=lambda rows,section,key:float(np.mean([r[section][key] for r in rows]))
    displacement=lambda rows:float(np.mean([r['fields']['current']['mean_km'] for r in rows]))
    current_control=float(np.mean([r['fields']['learned_velocity_effect_same_features']['relative_vector_change'] for r in trained]))
    endpoint_change=float(np.mean([r['endpoint_vector_comparison']['relative_change'] for r in trained]))
    stats=dict(initial_mean_displacement_km=displacement(initial),trained_mean_displacement_km=displacement(trained),
               endpoint_relative_vector_change=endpoint_change,
               learned_velocity_effect_same_features_relative=current_control,
               trained_correction_to_hidden_rms=avg(trained,'scalars','correction_to_hidden_rms'),
               trained_changed_hidden_fraction=avg(trained,'scalars','changed_hidden_fraction'),
               trained_effective_correction_rms=avg(trained,'scalars','effective_correction_rms'),
               trained_raw_correction_rms=avg(trained,'scalars','correction_rms'))
    repeats=json.loads((root/'repeatability.json').read_text())
    repeat_summary={}
    for case in ['baseline','advection','advection_off']:
        rr=[r for r in repeats['records'] if r['model']==case and r['variable']=='2m_temperature']
        values=np.array([r['rmse'] for r in rr])
        repeat_summary[case]=dict(rmse_mean=float(values.mean()),rmse_std=float(values.std()),
            rmse_min=float(values.min()),rmse_max=float(values.max()),
            repeat_field_rms=float(np.mean([r['rms_difference_from_first'] for r in rr[1:]])))
    stats['repeatability_2m_temperature']=repeat_summary
    (root/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    rows=list(csv.DictReader((root/'scores.csv').open()))
    lookup={(r['model'],r['variable'],r['level_hpa']):r for r in rows}
    cases=['pretrained','baseline_trained','advection_initial','advection_trained','advection_off','advection_reset_velocity']
    labels=['Pretrained at 1°','Direct fine-tuning','Initial adapter replay','Trained advection','Trained advection, zero correction','Trained advection, initial displacement weights']
    train_rows={}
    for name in ['baseline','advection']:
        train_rows[name]=[r for r in map(json.loads,(a.training_root/name/'metrics.jsonl').read_text().splitlines()) if 'val_loss' in r]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    for name,color,style in [('baseline','#2166ac','-'),('advection','#d6604d','--')]:
        rr=train_rows[name]
        axes[0].plot([r['update'] for r in rr],[r['val_loss'] for r in rr],style,color=color,label=name,lw=2)
    axes[0].set(yscale='log',xlabel='Fine-tuning update',ylabel='Normalized validation MSE',title='Both models learn; curves nearly overlap')
    axes[0].legend();axes[0].grid(alpha=.2)
    b=np.array([r['val_loss'] for r in train_rows['baseline']]);v=np.array([r['val_loss'] for r in train_rows['advection']])
    axes[1].plot([r['update'] for r in train_rows['baseline']],100*(v/b-1),color='#d6604d',marker='o',ms=3)
    axes[1].axhline(0,color='#666',lw=.8)
    axes[1].set(xlabel='Fine-tuning update',ylabel='Advection minus baseline (%)',title='Small differences throughout training')
    axes[1].grid(alpha=.2)
    fig.suptitle('Logged losses on the four fixed 2020 checkpoint-selection dates',fontsize=13)
    fig.savefig(root/'learning-curves.png',dpi=170);plt.close(fig)
    maps=[np.load(root/f'map-{label}.npz') for label in ['initial','trained']]
    lo=np.degrees(maps[0]['longitude']);la=np.degrees(maps[0]['latitude'])
    before=maps[0]['mean_distance_km'];after=maps[1]['mean_distance_km']
    vmax=max(before.max(),after.max());delta=after-before;limit=max(abs(delta.min()),abs(delta.max()))
    fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
    for ax,field,title,cmap,vmin,vmax_ in zip(axes,[before,after,delta],
            ['Before training: random departure field','After 1,000 updates','After minus before'],
            ['viridis','viridis','RdBu_r'],[0,0,-limit],[vmax,vmax,limit]):
        sc=ax.scatter(lo,la,c=field,s=3,cmap=cmap,vmin=vmin,vmax=vmax_,rasterized=True)
        ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=title)
        fig.colorbar(sc,ax=ax,label='Latent sampling distance (km equivalent)',shrink=.85)
    fig.suptitle('15 January 2020: mean departure distance across 16 latent modes\nSampling offsets are not physical wind speeds',fontsize=13)
    fig.savefig(root/'displacement-maps.png',dpi=160);plt.close(fig)
    report=['# Displacement and direct fine-tuning audit','',
        'Six forecast cases were compared on the same 12 monthly 2020 validation dates, at 1° and a six-hour horizon. '
        'Latent fields were measured on January, April, July and October dates. No training updates were run and '
        'no new test-year results were used for these diagnostics.','',
        '## Displacement before and after','',
        f'- Mean latent departure distance: **{stats["initial_mean_displacement_km"]:.2f} → {stats["trained_mean_displacement_km"]:.2f} km equivalent** '
        '(uniform mean over mesh nodes, 16 modes and four validation dates).',
        f'- The east/north displacement vectors changed by **{100*endpoint_change:.2f}%** in relative L2 norm, averaged over dates. '
        'This includes both backbone-feature changes and displacement-network learning.',
        f'- Holding the final backbone features fixed and swapping only initial versus trained displacement weights changes '
        f'the vectors by **{100*current_control:.2f}%** in relative L2 norm.',
        '- Initial displacements are random and nonzero. The initial output projection is zero, so they initially add no latent correction.',
        f'- The trained correction RMS is **{100*stats["trained_correction_to_hidden_rms"]:.5f}%** of backbone feature RMS.',
        f'- After materializing the actual BF16 decoder-input tensors, **{100*stats["trained_changed_hidden_fraction"]:.3f}%** of mesh-feature components change. '
        'This is the fraction of components whose represented value changes, not the fraction of correction energy retained.','',
        '![Displacement maps](displacement-maps.png)','',
        '## Direct fine-tuning and controlled forecast comparisons','',
        f'Direct fine-tuning changed {changes["direct_finetuning"]["changed"]:,} of '
        f'{changes["direct_finetuning"]["parameters"]:,} active parameter values '
        f'({100*changes["direct_finetuning"]["relative_delta"]:.3f}% relative L2 change). '
        f'The two trained backbones differ by {100*changes["trained_backbones_difference"]["relative_delta"]:.4f}% in relative L2 norm.','']
    for var,level,label,unit in [('2m_temperature','','2 m temperature','K'),('temperature','850','850 hPa temperature','K'),('geopotential','500','500 hPa geopotential','m²/s²')]:
        report += [f'### {label}', '',f'| Case | RMSE ({unit}) | ACC |','|---|---:|---:|']
        for case,label_ in zip(cases,labels):
            r=lookup[(case,var,level)]
            report.append(f'| {label_} | {float(r["rmse"]):.6f} | {float(r["acc"]):.7f} |')
        report.append('')
    report += ['The adapter-off comparison retains the trained advection backbone and zeros only the final lift matrix. '
               'The reset-displacement comparison retains that backbone, learned projection and learned lift, but restores '
               'the initial displacement linear weights/biases. These are evaluation-only ablations, not additional trained models.','',
               '![Validation learning curves](learning-curves.png)','',
               '## Repeatability check','',
               'Eight repetitions per case used identical 15 January 2020 inputs, weights and PRNG key. '
               'The current GPU/BF16 execution shows numerical variation between repeats. This does not establish '
               'BF16 casting alone as the cause; rounding itself is deterministic.','',
               '| Case | Mean 2 m temperature RMSE (K) | Repeat standard deviation (K) | Range (K) |',
               '|---|---:|---:|---:|']
    for case,r in repeat_summary.items():
        report.append(f'| {case} | {r["rmse_mean"]:.8f} | {r["rmse_std"]:.8f} | {r["rmse_min"]:.8f}–{r["rmse_max"]:.8f} |')
    report += ['',
               'The adapter-on/off RMSE gap is smaller than the per-date repeat variation measured here. '
               'The sample does not establish a meaningful incremental advection benefit. The direct-fine-tuning improvement is much larger. '
               'Recomputed baseline/advection scores differ slightly from the earlier benchmark, so the table above is the matched '
               'diagnostic rerun; it does not replace the original benchmark record.','',
               '## Reproducibility and limits','',
               'No epoch-zero adapter checkpoint was retained. Its parameters were reconstructed from the original pretrained '
               'backbone, the recorded seed and the exact training initializer; the Haiku shape and traversal order were checked. '
               f'The original initial validation loss was {status["initial_loss_recorded"]:.9f}; the replay produced '
               f'{status["initial_loss_reproduced"]:.9f}. This small BF16 numerical difference is disclosed; the replay is not a saved epoch-zero artifact.','',
               'Only initial and final adapter parameters are available, so the plots do not establish the displacement trajectory '
               'at intermediate updates. The learning curves use the actual saved validation logs at every 50 updates. '
               'Latent capture calls the same normalized single-step network inside the autoregressive wrapper; all physical '
               'forecast comparisons use the unchanged production rollout. Checkpoint hashes were checked against the frozen evaluation.','',
               '[All validation scores](scores.csv) · [Displacement statistics](displacement.json) · '
               '[Parameter changes](parameter_changes.json) · [Forecast differences](prediction_effects.json)','']
    (root/'report.md').write_text('\n'.join(report))
    print(json.dumps(stats,indent=2))
    for case in cases:
        r=lookup[(case,'2m_temperature','')];print(case,r['rmse'],r['acc'])


if __name__=='__main__':main()
