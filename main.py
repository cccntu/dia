import os
import torch
torch.set_float32_matmul_precision('high')
print(f'torch version: {torch.__version__}')
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
import  os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from dia.model import Dia
model = Dia.from_pretrained("ttj/dia-1.6b-safetensors", dtype="bf16")

import time

def generate(text: str, prof=None):
    tik = time.time()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model.generate(text, use_torch_compile=True, prof=prof, dtype="bfloat16",
                                compile_kwargs=dict(mode='max-autotune')
                            )
    print(f'{output.shape=}')
    step = output.shape[0]
    tok = time.time()
    print(f'time: {tok-tik}, step: {step}, steps/s: {step/(tok-tik)}')
    return output
# %%
# warmup
text = '[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices.'

output = generate(text)
output = generate(text)
output = generate(text)
# %%
from torch.profiler import profile, ProfilerActivity, schedule
sched = schedule(wait=0, warmup=0, active=20, repeat=1)

# Run 3 times with profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    schedule=sched,
) as prof:
    for _ in range(1):
        output = generate(text, prof=prof)
prof.export_chrome_trace("trace.json")
# %%
import soundfile as sf
import numpy as np

# Save output to mp3
output_path = "output.mp3"
sf.write(output_path, output, 44100)
print(f"Saved audio to {output_path}")

