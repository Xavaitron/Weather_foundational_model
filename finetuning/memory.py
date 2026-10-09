"""Conservative pre-execution memory check for a single explicitly selected GPU."""
import os
import subprocess
import jax


def estimate(memory):
    return (memory.argument_size_in_bytes + memory.output_size_in_bytes
            + memory.temp_size_in_bytes - memory.alias_size_in_bytes)


def guard(memory):
    if jax.devices()[0].platform != 'gpu':
        return
    selected = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected.startswith('GPU-') or ',' in selected:
        raise RuntimeError('Use scripts/run_finetune.sh to select one GPU by UUID')
    # A broken unrelated GPU on this server prevents NVML's UUID lookup; index
    # selection works. Verify the returned UUID before trusting the free bytes.
    index = os.environ.get('GRAPHCAST_GPU',selected)
    row = subprocess.check_output([
        'nvidia-smi','-i',index,'--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip()
    actual_uuid,free_text = [part.strip() for part in row.split(',')]
    if actual_uuid != selected:
        raise RuntimeError('Memory query GPU does not match CUDA_VISIBLE_DEVICES')
    free_mib = int(free_text)
    free = free_mib*1024**2
    required = estimate(memory)
    reserve = 2*1024**3
    if required+reserve > free:
        raise MemoryError(f'Compiled update needs about {required/1024**3:.1f} GiB plus '
                          f'2 GiB reserve; only {free/1024**3:.1f} GiB is currently free. '
                          'No update executed. Compiler estimates exclude some runtime overhead.')
