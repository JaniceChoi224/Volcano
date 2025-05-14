import sounddevice as sd
import scipy.io.wavfile as wav
import os
import asyncio
import requests
import json
import base64
import websockets
import uuid
import gzip
import copy
from pydub import AudioSegment
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv


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
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

spk_id = "S_r5Xp9i6s1"
emotion = "angry"
emotion_scale = 3


class ChatRequest(BaseModel):
    message: str
    path: str


class CharacterInfo(BaseModel):
    name: str
    gender: str
    age: str
    job: str
    relationship: str


def record_audio(filename: str, duration: int = 10, samplerate: int = 24000):
    """Record audio from the microphone."""
    filepath = os.path.join(AUDIO_SAVE_PATH, filename)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until finished
    wav.write(dirpath + filepath, samplerate, recording)
    return filepath


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


def convert_audio_to_wav(file_content: bytes, input_format: str) -> str:
    """
    Converts an in-memory audio file (bytes) to WAV format and saves it to a directory.

    Args:
        file_content (bytes): The input audio file content in bytes.
        input_format (str): The format of the input file (e.g., "mp3", "wav").

    Returns:
        str: The path to the saved WAV file.
    """
    # Convert the file content to an audio segment
    audio = AudioSegment.from_file(BytesIO(file_content), format=input_format)

    # Define output filename
    wav_filename = "voice_sample.wav"

    wav_file_path = os.path.join(AUDIO_SAVE_PATH, wav_filename)

    # Export to .wav format
    audio.export(dirpath + wav_file_path, format="wav")

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


def fill_template(character_info: CharacterInfo) -> str:
    # Read template
    filename = 'character_info_template.txt'
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
    filename = "character.json"
    filepath = os.path.join(CHAT_SAVE_PATH, filename)

    with open(dirpath + filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'], filepath
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