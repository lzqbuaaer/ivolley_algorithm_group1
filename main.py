from pathlib import Path

from bottle import Bottle, request

from src.enums.action import Action
from src.result.result import CommonResult
from src.G3.rules.rule import Rule
from src.scheduler import *
from src.utils.logger import Log

# for debug
debug = True

# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

app = Bottle()


@app.route('/cv', method='POST')  # 设置路由请求和http请求所处理的动作
def process():
    url = request.json['file_url']
    tag = request.json['tag']
    Log.warning(f"request.json = {request.json}")
    client_ip = request.environ.get('REMOTE_ADDR')
    Log.info("IP: " + client_ip + " FILE: " + url)
    try:
        result = solve(url, tag)
        Log.info(f"完成请求 {tag} + {url}")
        return result
    except Exception as e:
        Log.error(str(e))
        return CommonResult.fail(str(e))


def solve(url, tag):
    if Path(url).suffix[1:] in VID_FORMATS:
        mes, output_path = analyze_video(url, int(tag))
        ret = CommonResult.success("; ".join(mes), None, output_path)
        return ret
    elif Path(url).suffix[1:] in IMG_FORMATS:
        mes, output_path = analyze_image(url, int(tag))
        ret = CommonResult.success("; ".join(mes), None, output_path)
        return ret
    else:
        raise Exception("上传的文件不符合要求")


# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    Log.info("项目已启动")
    if debug:
        ret = solve(os.path.join('videos', 'spike2.mp4'), 1)
        print("ret = ", ret)
    else:
        app.run(host='127.0.0.1', port=5000)
