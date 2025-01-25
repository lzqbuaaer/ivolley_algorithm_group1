import json_tricks as json


class CommonResult():
    def __init__(self, code, message, data=None, outputPath=None):
        self.code = code
        self.message = message
        self.data = data
        self.outputPath = outputPath

    @classmethod
    def success(self, message, data, outputPath=None):
        return json.dumps(CommonResult(200, message, json.dumps(data), outputPath))

    @classmethod
    def fail(self, message):
        return json.dumps(CommonResult(500, message))