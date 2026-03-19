import json
import threading
import time
import torch
import os
from collections import defaultdict
from dataclasses import dataclass

class Schedule:
    def __init__(self, wait=0, warmup=0, active=1):
        self.wait = wait
        self.warmup = warmup
        self.active = active
    
    def phase(self, step: int) -> str:
        if step < self.wait:
            return "wait"
        elif step < self.wait + self.warmup:
            return "warmup"
        elif step < self.wait + self.warmup + self.active:
            return "active"
        else:
            return "inactive"

@dataclass
class TraceEvent:
    name: str
    cat: str
    ph: str
    ts: float
    dur: float
    pid: int
    tid: int


class Profile:
    def __init__(self, model, name="model", schedule=None, path="custom_profiler/trace.json"):
        self.model = model
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self._hooks = []

        # CPU timing
        self._fw_stack = {}
        self._bw_stack = {}

        # GPU timing (store events)
        self._fw_cuda = {}
        self._bw_cuda = {}

        self._step = 0
        self._pid = os.getpid()
        self._tid = threading.get_native_id()

        self._use_cuda = next(model.parameters()).is_cuda

        if hasattr(schedule, "phase") and callable(schedule.phase):
            print("Using custom schedule function")
            self._schedule_fn = schedule.phase
        else:
            self._schedule_fn = lambda step: "active"

        self._phase = "inactive"
        
        self._path = path

    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = full_name
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _register_hooks(self):
        for module in self.model.modules():
            self._hooks.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self._hooks.append(module.register_forward_hook(self._forward_post_hook))

            self._hooks.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self._hooks.append(module.register_full_backward_hook(self._backward_post_hook))


    def _forward_pre_hook(self, module, inputs):
        if not self._should_track_time():
            return

        cpu_start = self._now_us()
        self._fw_stack.setdefault(module, []).append(cpu_start)

        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._fw_cuda.setdefault(module, []).append([cpu_start, start, None])

    def _forward_post_hook(self, module, inputs, outputs):
        if not self._should_track_time():
            return

        # CPU
        start = self._fw_stack[module].pop()
        end = self._now_us()

        # GPU
        if self._use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._fw_cuda[module][-1][2] = end_event

        if not self._should_record():
            return

        self.events.append(TraceEvent(
            name=self.name_map.get(module, module.__class__.__name__),
            cat="forward_cpu",
            ph="X",
            ts=start,
            dur=end - start,
            pid=self._pid,
            tid=self._tid
        ))


    def _backward_pre_hook(self, module, grad_output):
        if not self._should_track_time():
            return

        cpu_start = self._now_us()
        self._bw_stack.setdefault(module, []).append(cpu_start)

        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._bw_cuda.setdefault(module, []).append([cpu_start, start, None])

    def _backward_post_hook(self, module, grad_input, grad_output):
        if not self._should_track_time():
            return

        start = self._bw_stack[module].pop()
        end = self._now_us()

        if self._use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._bw_cuda[module][-1][2] = end_event

        if not self._should_record():
            return

        self.events.append(TraceEvent(
            name=self.name_map.get(module, module.__class__.__name__),
            cat="backward_cpu",
            ph="X",
            ts=start,
            dur=end - start,
            pid=self._pid,
            tid=self._tid
        ))


    def __enter__(self):
        if self._use_cuda:
            torch.cuda.synchronize()
        
        self._register_hooks()
        self._phase = self._schedule_fn(self._step)
        return self

    def __exit__(self, type, value, traceback):
        for hook in self._hooks:
            hook.remove()

        if self._use_cuda:
            torch.cuda.synchronize()

            for module, pairs in self._fw_cuda.items():
                for cpu_start, start, end in pairs:
                    if start and end:
                        self.events.append(TraceEvent(
                            name=self.name_map.get(module, module.__class__.__name__),
                            cat="forward_gpu",
                            ph="X",
                            ts=cpu_start,
                            dur=start.elapsed_time(end) * 1000,
                            pid=self._pid,
                            tid=self._tid
                        ))

            for module, pairs in self._bw_cuda.items():
                for cpu_start, start, end in pairs:
                    if start and end:
                        self.events.append(TraceEvent(
                            name=self.name_map.get(module, module.__class__.__name__),
                            cat="backward_gpu",
                            ph="X",
                            ts=cpu_start,
                            dur=start.elapsed_time(end) * 1000,
                            pid=self._pid,
                            tid=self._tid
                        ))

        self._fw_stack.clear()
        self._bw_stack.clear()
        
        self.to_perfetto(path=self._path)


    def step(self):
        self._step += 1
        self._phase = self._schedule_fn(self._step)
        print(f"Step {self._step}, Phase: {self._phase}")

    def summary(self):
        print("Summary:")
        for e in self.events:
            print(e)

    def to_perfetto(self, path="custom_profiler/trace.json"):
        trace = {
            "traceEvents": [e.__dict__ for e in self.events],
            "displayTimeUnit": "ms"
        }
        if len(trace["traceEvents"]) == 0:
            return
        with open(path, "w") as f:
            json.dump(trace, f)

    def _now_us(self):
        return time.perf_counter_ns() / 1000.0

    def _should_track_time(self):
        return self._phase in ("warmup", "active")

    def _should_record(self):
        return self._phase == "active"
