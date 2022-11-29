# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from sanic import Sanic, response
import subprocess

import os
os.environ['HF_AUTH_TOKEN'] = "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"
os.environ['aws_id'] = "AKIASAKJNN7D2SAHBKX4"
os.environ['aws_secret'] = "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"

import app as user_src

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
# user_src.init_models()
from multiprocessing import set_start_method
# Create the http server app
server = Sanic("my_app")
try:
    set_start_method('spawn')
except:
    pass
# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json
    user_src.init_models()
    output = user_src.inference(model_inputs)

    return response.json(output)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8000, workers=1)