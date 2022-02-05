from fastapi import FastAPI, Request
from Routers import predictModel

app = FastAPI()


app.include_router(predictModel.router)


@app.get('/ping', tags=['Ping'])
def pingServer(request: Request):
    return "Application starts on " + request.client.host + ':' + str(request.client.port)
