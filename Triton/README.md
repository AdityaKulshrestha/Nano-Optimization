



## Setup
1. Create a virtual environment
    `uv venv --python=3.12`

2. Activate the virtual environment
    `source venv/bin/activate`

3. Install torch and triton
    `uv pip install torch --torch-backend=xpu/auto/cpu`

