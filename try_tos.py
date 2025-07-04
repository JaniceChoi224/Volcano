import os
import tos
from dotenv import load_dotenv
dirpath = os.path.dirname(__file__)


load_dotenv()  # defaults to .env in current dir

# 从环境变量获取 AK 和 SK 信息。
ak = os.getenv('TOS_ACCESS_KEY')
sk = os.getenv('TOS_SECRET_KEY')
# your endpoint 和 your region 填写Bucket 所在区域对应的Endpoint。# 以华北2(北京)为例，your endpoint 填写 tos-cn-beijing.volces.com，your region 填写 cn-beijing。
endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket_name = "voicecompanion2"
# 对象名称，例如 example_dir 下的 example_object.txt 文件，则填写为 example_dir/example_object.txt
object_key = "voice_sample.wav"
# 本地文件路径
file_name = dirpath + "/static/recordings/voice_sample.wav"
try:
    # 创建 TosClientV2 对象，对桶和对象的操作都通过 TosClientV2 实现
    client = tos.TosClientV2(ak, sk, endpoint, region)
    # 将本地文件上传到目标桶中
    # file_name为本地文件的完整路径。
    client.put_object_from_file(bucket_name, object_key, file_name)
except tos.exceptions.TosClientError as e:
    # 操作失败，捕获客户端异常，一般情况为非法请求参数或网络异常
    print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
except tos.exceptions.TosServerError as e:
    # 操作失败，捕获服务端异常，可从返回信息中获取详细错误信息
    print('fail with server error, code: {}'.format(e.code))
    # request id 可定位具体问题，强烈建议日志中保存
    print('error with request id: {}'.format(e.request_id))
    print('error with message: {}'.format(e.message))
    print('error with http code: {}'.format(e.status_code))
    print('error with ec: {}'.format(e.ec))
    print('error with request url: {}'.format(e.request_url))
except Exception as e:
    print('fail with unknown error: {}'.format(e))