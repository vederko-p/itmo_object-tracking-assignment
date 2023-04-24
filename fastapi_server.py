from fastapi import FastAPI, WebSocket
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from track_3 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np
#from trackers.strong.deep_sort import nn_matching
#from trackers.strong.deep_sort.detection import Detection
#from trackers.strong.deep_sort.tracker import Tracker as DeepSort
from trackers.strong.deep_sort import build_tracker
from utils.parser import get_config
from sklearn.metrics.pairwise import cosine_distances

app = FastAPI(title='Tracker assignment')
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
imgs = glob.glob('./static/imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
cfg = get_config(config_file="./configs/deep_sort.yaml")
deepsort = build_tracker(cfg, use_cuda=False)
print('Started')


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    '''
    detections = [x for x in el['data'] if len(x['bounding_box']) > 0]
    dets = [Detection(np.array(x['bounding_box']), 1.0, (1,2,3)) for x in detections]
    '''
    el['data'] = deepsort.update(el['data'], el['frame_id'])
    return el


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('index.html', context)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        #el = tracker_soft(el)
        # TODO: part 2
        el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Bye..')
