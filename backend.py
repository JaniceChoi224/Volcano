# import sounddevice as sd
# import scipy.io.wavfile as wav
import os
import requests
import json
import base64
import websockets
import uuid
import gzip
import copy
import tos
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
from models import *
import re


load_dotenv()  # defaults to .env in current dir

dirpath = os.path.dirname(__file__)

if not dirpath.__eq__(os.getcwd()):
    dirpath = os.getcwd()
    AUDIO_SAVE_PATH = "app/static/recordings"
    CHAT_SAVE_PATH = "app/static/chats"
    TEMPLATE_PATH = "app/static/templates"
else:
    AUDIO_SAVE_PATH = "static/recordings"
    CHAT_SAVE_PATH = "static/chats"
    TEMPLATE_PATH = "static/templates"

dirpath += "/"
os.makedirs(dirpath + AUDIO_SAVE_PATH, exist_ok=True)
os.makedirs(dirpath + CHAT_SAVE_PATH, exist_ok=True)
os.makedirs(dirpath + TEMPLATE_PATH, exist_ok=True)

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# spk_id = "S_r5Xp9i6s1"
spk_id = "zh_female_roumeinvyou_emo_v2_mars_bigtts"
emotion = "neutral"
emotion_scale = 3


# def record_audio(filename: str, duration: int = 10, samplerate: int = 24000):
#     """Record audio from the microphone."""
#     filepath = os.path.join(AUDIO_SAVE_PATH, filename)
#     recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
#     sd.wait()  # Wait until finished
#     wav.write(dirpath + filepath, samplerate, recording)
#     upload_audio(filename, dirpath + filepath)
#     return filepath


def check_file_exists(category: str, filename: str) -> bool:
    """
    Checks if the specified file exists in the directory.

    Args:
        category (str): The category where the file belong.
        filename (str): The filename of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if category == 'audio':
        file_path = os.path.join(AUDIO_SAVE_PATH, filename)
    elif category == 'templates':
        file_path = os.path.join(TEMPLATE_PATH, filename)
    elif category == 'chats':
        file_path = os.path.join(CHAT_SAVE_PATH, filename)
    return os.path.exists(dirpath + file_path)


def convert_audio_to_wav(file_content: bytes, input_format: str, filename: str) -> str:
    """
    Converts an in-memory audio file (bytes) to WAV format and saves it to a directory.

    Args:
        file_content (bytes): The input audio file content in bytes.
        input_format (str): The format of the input file (e.g., "mp3", "wav").
        filename (str):  The name of the output file.

    Returns:
        str: The path to the saved WAV file.
    """
    # Convert the file content to an audio segment
    audio = AudioSegment.from_file(BytesIO(file_content), format=input_format)

    wav_file_path = os.path.join(AUDIO_SAVE_PATH, filename)

    # Export to .wav format
    audio.export(dirpath + wav_file_path, format="wav", parameters=[
        "-ar", "24000",  # sample rate: 24000 Hz
        "-ac", "1",      # channels: 1 (mono)
        "-sample_fmt", "s16"  # sample format: 16-bit PCM
    ])

    upload_audio_tos(filename, dirpath + wav_file_path)
    print('Uploaded the .wav file to TOS')

    return wav_file_path


def voice_clone():
    ref_audio = AUDIO_SAVE_PATH + "/voice_sample.wav"
    train(appid=VOLCENGINE_APPID, token=VOLCENGINE_TOKEN, audio_path=ref_audio, spk_id=spk_id)
    get_status(appid=VOLCENGINE_APPID, token=VOLCENGINE_TOKEN, spk_id=spk_id)
    return spk_id


async def tts(text: str):
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(tts_submit(text, spk_id, emotion, emotion_scale))
    await tts_submit(text, spk_id, emotion, emotion_scale)
    return AUDIO_SAVE_PATH + "/test.mp3"


def load_character_info(json_path: str) -> CharacterInfo:
    """
    Loads a JSON file and parses it into a CharacterInfo instance.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        CharacterInfo: Parsed character information.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return CharacterInfo(**data)


def fill_template(character_info: CharacterInfo) -> str:
    # Read template
    filename = 'initiate_chat_template.txt'
    filepath = os.path.join(TEMPLATE_PATH, filename)
    with open(dirpath + filepath, "r", encoding="utf-8") as f:
        template_content = f.read()

    # Replace placeholders
    filled_content = template_content.replace("*NAME_PLACEHOLDER*", character_info.name)
    filled_content = filled_content.replace("*GENDER_PLACEHOLDER*", character_info.gender)
    filled_content = filled_content.replace("*AGE_PLACEHOLDER*", character_info.age)
    filled_content = filled_content.replace("*JOB_PLACEHOLDER*", character_info.job)
    filled_content = filled_content.replace("*RELATIONSHIP_PLACEHOLDER*", character_info.relationship)

    return filled_content


def fill_daily_report_template(chatfname: str) -> str:
    # Read template
    filename = 'daily_report_template.txt'
    filepath = os.path.join(TEMPLATE_PATH, filename)
    with open(dirpath + filepath, "r", encoding="utf-8") as f:
        template_content = f.read()

    chat_path = os.path.join(CHAT_SAVE_PATH, chatfname)
    with open(dirpath + chat_path, "r", encoding="utf-8") as f:
        chat_content = f.read()

    # Replace placeholders
    filled_content = template_content.replace("*CHAT_PLACEHOLDER*", chat_content)

    return filled_content


def save_chat_txt(userfnames: UserFNames) -> None:
    """
    Extracts messages from a JSON file and saves them in a readable format to a text file.

    Each line in the output file will be: sender: message_text

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path to the output text file.
    """
    chat_json_path = os.path.join(CHAT_SAVE_PATH, userfnames.chat_json_fname)
    with open(dirpath + chat_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', [])
    chat_txt_path = os.path.join(CHAT_SAVE_PATH, userfnames.chat_txt_fname)
    with open(dirpath + chat_txt_path, 'w', encoding='utf-8') as f:
        for message in messages:
            role = message.get('role')
            content = message.get('content')
            if role and content:
                f.write(f"{role}: {content}\n")


def save_report_txt(userfnames: UserFNames) -> None:
    """
    Extracts messages from a JSON file and saves them in a readable format to a text file.

    Each line in the output file will be: sender: message_text

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path to the output text file.
    """
    report_json_path = os.path.join(CHAT_SAVE_PATH, userfnames.report_json_fname)
    with open(dirpath + report_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', [])
    report_txt_path = report_json_path = os.path.join(CHAT_SAVE_PATH, userfnames.report_txt_fname)
    with open(dirpath + report_txt_path, 'w', encoding='utf-8') as f:
        for message in messages:
            print(message)
            f.write(message)


def initiate_query_deepseek(character_info: CharacterInfo) -> str:
    filled_text = fill_template(character_info)

    dict_data = {
    "messages": [
        {
        "content": filled_text,
        "role": "system"
        },
        {
        "content": "请开始对话。",
        "role": "user"
        }
    ],
    "model": "deepseek-chat",
    "frequency_penalty": 0,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "response_format": {
        "type": "text"
    },
    "stop": None,
    "stream": False,
    "stream_options": None,
    "temperature": 1,
    "top_p": 1,
    "tools": None,
    "tool_choice": "none",
    "logprobs": False,
    "top_logprobs": None
    }

    payload = json.dumps(dict_data)

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    response = requests.request("POST", DEEPSEEK_API_URL, headers=headers, data=payload)
    dict_data['messages'].append(response.json()['choices'][0]['message'])

    # filename = f"{character_info.name}.json"
    filename = "chat.json"
    filepath = os.path.join(CHAT_SAVE_PATH, filename)

    with open(dirpath + filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'], filepath
    else:
        return f"Error from DeepSeek: {response.text}"


def load_daily_report(file_path: str) -> DailyReport:
    mapping = {
        '综合得分': ('overall_score', 'overall_comments'),
        '心理健康': ('mental_health_score', 'mental_health_comments'),
        '兴趣与需求': ('interests_needs_score', 'interests_needs_comments'),
        '家庭联结': ('family_connection_score', 'family_connection_comments'),
        '安全认知': ('safety_awareness_score', 'safety_awareness_comments'),
        '身体健康': ('physical_health_score', 'physical_health_comments'),
        '生活状态': ('living_conditions_score', 'living_conditions_comments'),
    }

    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for key, (score_key, comment_key) in mapping.items():
                if line.startswith(key):
                    match = re.search(r'(\d+)[，,]\s*评语[:：](.*)', line)
                    if match:
                        data[score_key] = int(match.group(1))
                        data[comment_key] = match.group(2).strip()
                    break
            else:
                if line.startswith('综合建议'):
                    data['overall_comments'] = line.split('：', 1)[-1].strip()

    return DailyReport(**data)


def get_daily_report_deepseek(userfnames: UserFNames) -> DailyReport:
    save_chat_txt(userfnames)
    print(f'User chat successfully saved to {userfnames.chat_txt_fname}.')
    filled_text = fill_daily_report_template(userfnames.chat_txt_fname)

    dict_data = {
    "messages": [
        {
        "content": filled_text,
        "role": "user"
        }
    ],
    "model": "deepseek-chat",
    "frequency_penalty": 0,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "response_format": {
        "type": "text"
    },
    "stop": None,
    "stream": False,
    "stream_options": None,
    "temperature": 1,
    "top_p": 1,
    "tools": None,
    "tool_choice": "none",
    "logprobs": False,
    "top_logprobs": None
    }

    payload = json.dumps(dict_data)

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    response = requests.request("POST", DEEPSEEK_API_URL, headers=headers, data=payload)
    dict_data['messages'].append(response.json()['choices'][0]['message'])
    

    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']

        report_txt_path = os.path.join(CHAT_SAVE_PATH, userfnames.report_txt_fname)
        with open(dirpath + report_txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'User daily report successfully saved to {userfnames.report_txt_fname}.')
        report = load_daily_report(report_txt_path)
        return report
    else:
        return f"Error from DeepSeek: {response.text}"


def query_deepseek(message: str, history_filepath: str) -> str:
    with open(dirpath + history_filepath) as f:
        dict_data = json.load(f)

    dict_data['messages'].append(
        {
            "content": message,
            "role": "user" 
        }
    )

    payload = json.dumps(dict_data)

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    response = requests.request("POST", DEEPSEEK_API_URL, headers=headers, data=payload)

    dict_data['messages'].append(response.json()['choices'][0]['message'])

    with open(dirpath + history_filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return f"Error from DeepSeek: {response.text}"
    

'''
VOLCENGINE
'''
MESSAGE_TYPES = {11: "audio-only server response", 12: "frontend server response", 15: "error message from server"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "no sequence number", 1: "sequence number > 0",
                               2: "last message from server (seq < 0)", 3: "sequence number < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

VOLCENGINE_APPID = os.getenv("VOLCENGINE_APPID")
VOLCENGINE_TOKEN = os.getenv("VOLCENGINE_TOKEN")
VOLCENGINE_HOST = os.getenv("VOLCENGINE_HOST")

with open("voice_type_list.json") as f:
    VOICE_TYPE_LIST = json.load(f)

# version: b0001 (4 bits)
# header size: b0001 (4 bits)
# message type: b0001 (Full client request) (4bits)
# message type specific flags: b0000 (none) (4bits)
# message serialization method: b0001 (JSON) (4 bits)
# message compression: b0001 (gzip) (4bits)
# reserved data: 0x00 (1 byte)
default_header = bytearray(b'\x11\x10\x11\x00')

request_json = {
    "app": {
        "appid": VOLCENGINE_APPID,
        "token": "access_token",
        "cluster": "xxx"
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": "xxx",
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
        "enable_emotion": True,
        "emotion": "xxx",
        "emotion_scale": 1.0
    },
    "request": {
        "reqid": "xxx",
        "text": "xxx",
        "text_type": "plain",
        "operation": "xxx"
    }
}


'''
Voice clone
'''
def train(appid, token, audio_path, spk_id):
    url = f"https://{VOLCENGINE_HOST}/api/v1/mega_tts/audio/upload"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    encoded_data, audio_format = encode_audio_file(audio_path)
    audios = [{"audio_bytes": encoded_data, "audio_format": audio_format}]
    data = {"appid": appid, "speaker_id": spk_id, "audios": audios, "source": 2,"language": 0, "model_type": 1}
    response = requests.post(url, json=data, headers=headers)
    print("status code = ", response.status_code)
    if response.status_code != 200:
        raise Exception("train请求错误:" + response.text)
    print("headers = ", response.headers)
    print(response.json())


def get_status(appid, token, spk_id):
    url = f"https://{VOLCENGINE_HOST}/api/v1/mega_tts/status"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    body = {"appid": appid, "speaker_id": spk_id}
    response = requests.post(url, headers=headers, json=body)
    print(response.json())


def encode_audio_file(file_path):
    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        encoded_data = str(base64.b64encode(audio_data), "utf-8")
        audio_format = os.path.splitext(file_path)[1][1:]  # 获取文件扩展名作为音频格式
        return encoded_data, audio_format


'''
TTS
'''
async def tts_submit(text, spk_id, emotion, emotion_scale):
    url = f"wss://{VOLCENGINE_HOST}/api/v1/tts/ws_binary"
    voice_type = VOICE_TYPE_LIST[spk_id]

    cluster = voice_type["cluster"]
    submit_request_json = copy.deepcopy(request_json)
    submit_request_json["app"]["cluster"] = cluster
    submit_request_json["audio"]["voice_type"] = spk_id
    if emotion in voice_type["emotions"]:
        submit_request_json["audio"]["emotion"] = emotion
        submit_request_json["audio"]["emotion_scale"] = emotion_scale
    submit_request_json["request"]["reqid"] = str(uuid.uuid4())
    submit_request_json["request"]["text"] = text
    submit_request_json["request"]["operation"] = "submit"

    payload_bytes = str.encode(json.dumps(submit_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
    full_client_request.extend(payload_bytes)  # payload
    print("\n------------------------ 'submit' -------------------------")
    print("request json: ", submit_request_json)
    print("\nrequest bytes: ", full_client_request)
    filepath = os.path.join(AUDIO_SAVE_PATH, "test.mp3")
    file_to_save = open(filepath, "wb")
    header = {"Authorization": f"Bearer; {VOLCENGINE_TOKEN}"}
    async with websockets.connect(url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        while True:
            res = await ws.recv()
            done = parse_response(res, file_to_save)
            if done:
                file_to_save.close()
                break
        print("\nclosing the connection...")


def parse_response(res, file):
    print("--------------------------- response ---------------------------")
    # print(f"response raw bytes: {res}")
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size*4]
    payload = res[header_size*4:]
    print(f"            Protocol version: {protocol_version:#x} - version {protocol_version}")
    print(f"                 Header size: {header_size:#x} - {header_size * 4} bytes ")
    print(f"                Message type: {message_type:#x} - {MESSAGE_TYPES[message_type]}")
    print(f" Message type specific flags: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS[message_type_specific_flags]}")
    print(f"Message serialization method: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS[serialization_method]}")
    print(f"         Message compression: {message_compression:#x} - {MESSAGE_COMPRESSIONS[message_compression]}")
    print(f"                    Reserved: {reserved:#04x}")
    if header_size != 1:
        print(f"           Header extensions: {header_extensions}")
    if message_type == 0xb:  # audio-only server response
        if message_type_specific_flags == 0:  # no sequence number as ACK
            print("                Payload size: 0")
            return False
        else:
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload = payload[8:]
            print(f"             Sequence number: {sequence_number}")
            print(f"                Payload size: {payload_size} bytes")
        file.write(payload)
        if sequence_number < 0:
            return True
        else:
            return False
    elif message_type == 0xf:
        code = int.from_bytes(payload[:4], "big", signed=False)
        msg_size = int.from_bytes(payload[4:8], "big", signed=False)
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        error_msg = str(error_msg, "utf-8")
        print(f"          Error message code: {code}")
        print(f"          Error message size: {msg_size} bytes")
        print(f"               Error message: {error_msg}")
        return True
    elif message_type == 0xc:
        msg_size = int.from_bytes(payload[:4], "big", signed=False)
        payload = payload[4:]
        if message_compression == 1:
            payload = gzip.decompress(payload)
        print(f"            Frontend message: {payload}")
    else:
        print("undefined message type!")
        return True


# 从环境变量获取 AK 和 SK 信息。
TOS_ACCESS_KEY = os.getenv('TOS_ACCESS_KEY')
TOS_SECRET_KEY = os.getenv('TOS_SECRET_KEY')
# your endpoint 和 your region 填写Bucket 所在区域对应的Endpoint。# 以华北2(北京)为例，your endpoint 填写 tos-cn-beijing.volces.com，your region 填写 cn-beijing。
endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket_name = "voicecompanion2"


def upload_audio_tos(object_key, file_name):
    try:
        # 创建 TosClientV2 对象，对桶和对象的操作都通过 TosClientV2 实现
        client = tos.TosClientV2(TOS_ACCESS_KEY, TOS_SECRET_KEY, endpoint, region)
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


'''
STT
'''
def stt_submit(file_url):

    submit_url = f"https://{VOLCENGINE_HOST}/api/v3/auc/bigmodel/submit"

    task_id = str(uuid.uuid4())

    headers = {
        "X-Api-App-Key": VOLCENGINE_APPID,
        "X-Api-Access-Key": VOLCENGINE_TOKEN,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1"
    }

    request = {
        "user": {
            "uid": "fake_uid"
        },
        "audio": {
            "url": file_url,
            "format": "wav",
            "codec": "raw",
            "rate": 24000,
            "bits": 16,
            "channel": 1
        },
        "request": {
            "model_name": "bigmodel",
            # "enable_itn": True,
            # "enable_punc": True,
            # "enable_ddc": True,
            "show_utterances": True,
            # "enable_channel_split": True,
            # "vad_segment": True,
            # "enable_speaker_info": True,
            "corpus": {
                # "boosting_table_name": "test",
                "correct_table_name": "",
                "context": ""
            }
        }
    }
    print(f'Submit task id: {task_id}')
    response = requests.post(submit_url, data=json.dumps(request), headers=headers)
    if 'X-Api-Status-Code' in response.headers and response.headers["X-Api-Status-Code"] == "20000000":
        print(f'Submit task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Submit task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        print(f'Submit task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
        return task_id, x_tt_logid
    else:
        print(f'Submit task failed and the response headers are: {response.headers}')
        exit(1)
    return task_id


def stt_query(task_id, x_tt_logid):
    query_url = f"https://{VOLCENGINE_HOST}/api/v3/auc/bigmodel/query"

    headers = {
        "X-Api-App-Key": VOLCENGINE_APPID,
        "X-Api-Access-Key": VOLCENGINE_TOKEN,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid  # 固定传递 x-tt-logid
    }

    response = requests.post(query_url, json.dumps({}), headers=headers)

    if 'X-Api-Status-Code' in response.headers:
        print(f'Query task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Query task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        print(f'Query task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
    else:
        print(f'Query task failed and the response headers are: {response.headers}')
        exit(1)
    return response


def stt(filename: str):
    import time
    from fastapi import HTTPException
    file_url = f"https://voicecompanion2.tos-cn-beijing.volces.com/{filename}"
    task_id, x_tt_logid = stt_submit(file_url)
    # while True:
    for _ in range(30):  # max 30 retries
        query_response = stt_query(task_id, x_tt_logid)
        code = query_response.headers.get('X-Api-Status-Code', "")
        if code == '20000000':  # task finished
            print(query_response.json())
            print("SUCCESS!")
            return query_response.json()['result']['text']
        elif code != '20000001' and code != '20000002':  # task failed
            print("FAILED!")
            # exit(1)
            raise HTTPException(status_code=500, detail="STT task failed")
    time.sleep(1)
    raise HTTPException(status_code=504, detail="STT timed out")