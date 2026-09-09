# Design and interfaces

## Pipeline boundaries

The implementation separates concerns into dependency-light modules:

- `config.py`: early validation of user-controlled model, threshold, and path values
- `device.py`: explicit accelerator policy
- `media.py`: deterministic still, sequence, and video lifecycle
- `models.py`: official weight construction, BGR preprocessing, and output filtering
- `pipeline.py`: inference ownership and synchronized timing
- `types.py`: immutable internal detection and process-result types
- `annotation.py`: OpenCV rendering without mutating the source frame
- `ros_conversion.py`: Jazzy messages and diagnostics
- `metrics.py`: bounded mean, median, nearest-rank p95, count, and throughput
- `evaluation.py`: small-manifest AP50 evaluation

This boundary lets unit tests exercise geometry, preprocessing, filtering, and
metrics without a ROS daemon or model download. The smoke and graph checks then
verify the real integration path.

## Jazzy message mapping

The locally installed `vision_msgs` 4.1.1 schema was inspected with
`ros2 interface show`. Each internal XYXY detection maps as follows:

| Internal value | Jazzy field |
|---|---|
| source header | array and per-detection `header` |
| label string | `results[0].hypothesis.class_id` |
| score | `results[0].hypothesis.score` |
| `(x1+x2)/2` | `bbox.center.position.x` |
| `(y1+y2)/2` | `bbox.center.position.y` |
| `x2-x1` | `bbox.size_x` |
| `y2-y1` | `bbox.size_y` |
| axis-aligned box | `bbox.center.theta = 0` |

Per-frame detection IDs are stable only within one output message; tracking is
out of scope and no cross-frame identity is claimed.

## Timing semantics

- **Inference latency**: synchronized model call under `torch.inference_mode()`.
- **Processing latency**: preprocessing, transfer, inference, synchronization, output transfer, and filtering.
- **End-to-end ROS latency**: perception callback clock minus input header stamp.
- **Approximate throughput**: inverse of mean processing latency over the bounded window.

Annotation, ROS message construction, and publication occur after the reported
processing interval. ROS end-to-end timing naturally includes upstream queueing
but is valid only when clocks share a time domain.

## Failure behavior

Configuration and model-loading failures stop startup with actionable errors.
Individual malformed frames, conversion errors, and recoverable inference
errors are logged and dropped so later frames can proceed. Explicit CUDA fails
when unavailable; `auto` is the only mode allowed to fall back to CPU. Media EOF
stops a non-looping publisher cleanly, while invalid or undecodable media fails
before publication.
