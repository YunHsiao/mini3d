# mini3d
Simple scanline renderer in one C file

Features
===
* One single C file
* No dependencies
* Left-handed coordinate system, WORLD / VIEW / PROJECTION model
* CVV culling
* maximum 1024 x 1024 texture support
* depth buffer
* perspective-corrent texture mapping
* accurate edge calculation
* well-modulized, clean structure

Changes
===
* 360-degree camera control
* backfacing culling, near plane clipping
* point, line, color, texture render state
* load model from text file
* Bilinear filtered texture sampling
* per-pixel Blinn-Phong illumination
* simple shadow mapping
* flexible vertex format

Controls
===
|    Keys    |    Effects    |
|    :---:   |     :---:     |
| WSADQE     | move camera   |
| ↑ ↓ ← →    | rotate camera |
| numpad 1 2 | rotate model  |
| numpad 4 5 | incr/decr shadowmap precision |
| numpad 0 | camera focus to origin |
| N          | display model normal |
| F          | display wireframe model |
| ESC        | quit |
