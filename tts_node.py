import os,sys
from .ft_node import SoVITS_weight_root,GPT_weight_root,\
    pretrained_sovits_name,pretrained_gpt_name,work_path,\
        now_dir,gsv_path,models_dir,output_dir

sys.path.append(now_dir)
sys.path.append(gsv_path)
from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        try:
            for name in os.listdir(os.path.join(work_path,path)):
                if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
        except:
            pass
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        try:
            for name in os.listdir(os.path.join(work_path,path)):
                if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
        except:
            pass

    return SoVITS_names, GPT_names

dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v2

import re
import torch
import numpy as np
from time import time as ttime
import cuda_malloc
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

from module.models import SynthesizerTrn
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# import GPT_SoVITS.utils as utils
# def change_sovits_weights(sovits_path):
#     global vq_model, hps, version, dict_language
#     comfyui_utils = sys.modules['utils']
#     sys.modules['utils'] = utils
#     dict_s2 = torch.load(sovits_path, map_location="cpu")
#     sys.modules['utils'] = comfyui_utils
#     hps = dict_s2["config"]
#     hps = DictToAttrRecursive(hps)
#     hps.model.semantic_frame_rate = "25hz"
#     if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
#         hps.model.version = "v1"
#     else:
#         hps.model.version = "v2"
#     version = hps.model.version
#     # print("sovits版本:",hps.model.version)
#     vq_model = SynthesizerTrn(
#         hps.data.filter_length // 2 + 1,
#         hps.train.segment_size // hps.data.hop_length,
#         n_speakers=hps.data.n_speakers,
#         **hps.model
#     )
#     if ("pretrained" not in sovits_path):
#         del vq_model.enc_q
#     if is_half == True:
#         vq_model = vq_model.half().to(device)
#     else:
#         vq_model = vq_model.to(device)
#     vq_model.eval()
#     print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
#     dict_language = dict_language_v1 if version =='v1' else dict_language_v2
#     '''
#     with open("./weight.json")as f:
#         data=f.read()
#         data=json.loads(data)
#         data["SoVITS"][version]=sovits_path
#     with open("./weight.json","w")as f:f.write(json.dumps(data))
#     '''

_ = [[], []]
for i in range(4):
    tmp_gpt_name = os.path.join(models_dir, pretrained_gpt_name[i])
    tmp_sovits_name = os.path.join(models_dir, pretrained_sovits_name[i])
    if os.path.exists(tmp_gpt_name):
        _[0].append(tmp_gpt_name)
    if os.path.exists(tmp_sovits_name):
        _[-1].append(tmp_sovits_name)
pretrained_gpt_name, pretrained_sovits_name = _

path_sovits_v3 = pretrained_sovits_name[2]
path_sovits_v4 = pretrained_sovits_name[3]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)


from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3,Generator
from peft import LoraConfig, get_peft_model

v3v4set = {"v3", "v4"}

import GPT_SoVITS.utils as utils
def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    global vq_model, hps, version, model_version, dict_language, if_lora_v3
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(sovits_path,version, model_version, if_lora_v3)
    is_exist=is_exist_s2gv3 if model_version=="v3"else is_exist_s2gv4
    if if_lora_v3 == True and is_exist == False:
        info = "GPT_SoVITS/pretrained_models/s2Gv3.pth" + i18n("SoVITS %s 底模缺失，无法加载相应 LoRA 权重"%model_version)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {"__type__": "update", "value": i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("中文")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        # yield (
        #     {"__type__": "update", "choices": list(dict_language.keys())},
        #     {"__type__": "update", "choices": list(dict_language.keys())},
        #     prompt_text_update,
        #     prompt_language_update,
        #     text_update,
        #     text_language_update,
        #     {"__type__": "update", "visible": visible_sample_steps, "value": 32 if model_version=="v3"else 8,"choices":[4, 8, 16, 32,64,128]if model_version=="v3"else [4, 8, 16, 32]},
        #     {"__type__": "update", "visible": visible_inp_refs},
        #     {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
        #     {"__type__": "update", "visible": True if model_version =="v3" else False},
        #     {"__type__": "update", "value": i18n("模型加载中，请等待"), "interactive": False},
        # )

    comfyui_utils = sys.modules['utils']
    sys.modules['utils'] = utils
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version not in v3v4set:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        model_version = version
    else:
        hps.model.version=model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G"%model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version,lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()
    sys.modules['utils'] = comfyui_utils

    # yield (
    #     {"__type__": "update", "choices": list(dict_language.keys())},
    #     {"__type__": "update", "choices": list(dict_language.keys())},
    #     prompt_text_update,
    #     prompt_language_update,
    #     text_update,
    #     text_language_update,
    #     {"__type__": "update", "visible": visible_sample_steps, "value":32 if model_version=="v3"else 8,"choices":[4, 8, 16, 32,64,128]if model_version=="v3"else [4, 8, 16, 32]},
    #     {"__type__": "update", "visible": visible_inp_refs},
    #     {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
    #     {"__type__": "update", "visible": True if model_version =="v3" else False},
    #     {"__type__": "update", "value": i18n("合成语音"), "interactive": True},
    # )
    # with open("./weight.json") as f:
    #     data = f.read()
    #     data = json.loads(data)
    #     data["SoVITS"][version] = sovits_path
    # with open("./weight.json", "w") as f:
    #     f.write(json.dumps(data))


from AR.models.t2s_lightning_module import Text2SemanticLightningModule

# def change_gpt_weights(gpt_path):
#     global hz, max_sec, t2s_model, config
#     hz = 50
#     dict_s1 = torch.load(gpt_path, map_location="cpu")
#     config = dict_s1["config"]
#     max_sec = config["data"]["max_sec"]
#     t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
#     t2s_model.load_state_dict(dict_s1["weight"])
#     if is_half == True:
#         t2s_model = t2s_model.half()
#     t2s_model = t2s_model.to(device)
#     t2s_model.eval()
#     total = sum([param.nelement() for param in t2s_model.parameters()])
#     print("Number of parameter: %.2fM" % (total / 1e6))
#     '''
#     with open("./weight.json")as f:
#         data=f.read()
#         data=json.loads(data)
#         data["GPT"][version]=gpt_path
#     with open("./weight.json","w")as f:f.write(json.dumps(data))
#     '''

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    # with open("./weight.json") as f:
    #     data = f.read()
    #     data = json.loads(data)
    #     data["GPT"][version] = gpt_path
    # with open("./weight.json", "w") as f:
    #     f.write(json.dumps(data))

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

from text import chinese,cleaned_text_to_sequence
from text.cleaner import clean_text
import LangSegment
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio

def get_spepc(hps, audio):
    # audio = load_audio(filename, int(hps.data.sampling_rate))
    # audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert

def get_phones_and_bert(text,language,version):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
    dtype=torch.float16 if is_half == True else torch.float32
    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def get_tts_wav(
    ref_wav,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("不切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    sample_steps=16,
    if_freeze=False,
    inp_refs=None,
    if_sr=False,
    pause_second=0.3,
):
    dtype = torch.float16 if is_half == True else torch.float32

    t = []
    if len(prompt_text) == 0:
        ref_free = True
    else:
        ref_free = False

    if model_version in v3v4set:
        ref_free = False  # s2v3暂不支持ref_free

    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    if not ref_free:
        with torch.no_grad():
            # wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            # if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            #     gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
            #     raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            # wav16k = torch.from_numpy(wav16k)
            wav16k = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)(ref_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    ###s2v3暂不支持ref_free
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        # if i_text in cache and if_freeze == True:
        #     pred_semantic = cache[i_text]
        # else:
        #     with torch.no_grad():
        #         pred_semantic, idx = t2s_model.model.infer_panel(
        #             all_phoneme_ids,
        #             all_phoneme_len,
        #             None if ref_free else prompt,
        #             bert,
        #             # prompt_phone_len=ph_offset,
        #             top_k=top_k,
        #             top_p=top_p,
        #             temperature=temperature,
        #             early_stop_num=hz * max_sec,
        #         )
        #         pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        #         cache[i_text] = pred_semantic
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        t3 = ttime()
        ###v3不存在以下逻辑和inp_refs
        if model_version not in v3v4set:
            refers = []
            # if inp_refs:
            #     for path in inp_refs:
            #         try:
            #             refer = get_spepc(hps, path.name).to(dtype).to(device)
            #             refers.append(refer)
            #         except:
            #             traceback.print_exc()
            # if len(refers) == 0:
            #     refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
            dtype=torch.float16 if is_half == True else torch.float32
            if(len(refers)==0):refers = [get_spepc(hps, ref_wav).to(dtype).to(device)]
            # audio = vq_model.decode(
            #     pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
            # )[0][0]  # .cpu().detach().numpy()
            audio = vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
            ).cpu().detach().numpy()[0, 0]
        else:
            # refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)
            refer = get_spepc(hps, ref_wav).to(device).to(dtype)
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            # print(11111111, phoneme_ids0, phoneme_ids1)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            # ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_wav
            if ref_audio.dim() == 1:
                ref_audio = ref_audio.unsqueeze(0)
            sr = 16000
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            tgt_sr=24000 if model_version=="v3"else 32000
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr,tgt_sr)
            # print("ref_audio",ref_audio.abs().mean())
            mel2 = mel_fn(ref_audio)if model_version=="v3"else mel_fn_v4(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            Tref=468 if model_version=="v3"else 500
            Tchunk=934 if model_version=="v3"else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            cfm_resss = []
            idx = 0
            while 1:
                fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(
                    fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
                )
                cfm_res = cfm_res[:, :, mel2.shape[2] :]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cfm_res = torch.cat(cfm_resss, 2)
            cfm_res = denorm_spec(cfm_res)
            if model_version=="v3":
                if bigvgan_model == None:
                    init_bigvgan()
            else:#v4
                if hifigan_model == None:
                    init_hifigan()
            vocoder_model=bigvgan_model if model_version=="v3"else hifigan_model
            with torch.inference_mode():
                wav_gen = vocoder_model(cfm_res)
                # audio = wav_gen[0][0]  # .cpu().detach().numpy()
                audio = wav_gen.cpu().detach().numpy()[0, 0]
        # max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        # audio_opt.append(zero_wav_torch)  # zero_wav
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))

    return torch.Tensor(np.concatenate(audio_opt, 0)).unsqueeze(0)

    # audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
    # if model_version in {"v1","v2"}:opt_sr=32000
    # elif model_version=="v3":opt_sr=24000
    # else:opt_sr=48000#v4
    # if if_sr == True and opt_sr == 24000:
    #     print(i18n("音频超分中"))
    #     audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
    #     max_audio = np.abs(audio_opt).max()
    #     if max_audio > 1:
    #         audio_opt /= max_audio
    # else:
    #     audio_opt = audio_opt.cpu().detach().numpy()
    # yield opt_sr, (audio_opt * 32767).astype(np.int16)


# def get_tts_wav(ref_wav, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, speed=1, sample_steps=20):
#     t= []
#     if len(prompt_text) == 0:
#         ref_free = True
#     else:
#         ref_free = False
#     t0 = ttime()
#     prompt_language = dict_language[prompt_language]
#     text_language = dict_language[text_language]
#     print(f"prompt_language:{prompt_language}")
#     print(f"text_language:{text_language}")
#     if not ref_free:
#         prompt_text = prompt_text.strip("\n")
#         if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
#         print(i18n("实际输入的参考文本:"), prompt_text)
#     text = text.strip("\n")
#     if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
#
#     print(i18n("实际输入的目标文本:"), text)
#     zero_wav = np.zeros(
#         int(hps.data.sampling_rate * 0.3),
#         dtype=np.float16 if is_half == True else np.float32,
#     )
#     if not ref_free:
#         with torch.no_grad():
#             '''
#             wav16k, sr = librosa.load(ref_wav_path, sr=16000)
#             if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
#                 gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
#                 raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
#             wav16k = torch.from_numpy(wav16k)
#             '''
#             wav16k = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)(ref_wav)
#             zero_wav_torch = torch.from_numpy(zero_wav)
#             if is_half == True:
#                 wav16k = wav16k.half().to(device)
#                 zero_wav_torch = zero_wav_torch.half().to(device)
#             else:
#                 wav16k = wav16k.to(device)
#                 zero_wav_torch = zero_wav_torch.to(device)
#             wav16k = torch.cat([wav16k, zero_wav_torch])
#             ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
#                 "last_hidden_state"
#             ].transpose(
#                 1, 2
#             )  # .float()
#             codes = vq_model.extract_latent(ssl_content)
#             prompt_semantic = codes[0, 0]
#             prompt = prompt_semantic.unsqueeze(0).to(device)
#
#     t1 = ttime()
#     t.append(t1-t0)
#
#     if (how_to_cut == i18n("凑四句一切")):
#         text = cut1(text)
#     elif (how_to_cut == i18n("凑50字一切")):
#         text = cut2(text)
#     elif (how_to_cut == i18n("按中文句号。切")):
#         text = cut3(text)
#     elif (how_to_cut == i18n("按英文句号.切")):
#         text = cut4(text)
#     elif (how_to_cut == i18n("按标点符号切")):
#         text = cut5(text)
#     while "\n\n" in text:
#         text = text.replace("\n\n", "\n")
#     print(i18n("实际输入的目标文本(切句后):"), text)
#     texts = text.split("\n")
#     texts = process_text(texts)
#     texts = merge_short_text_in_array(texts, 5)
#     audio_opt = []
#     if not ref_free:
#         phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)
#
#     for i_text,text in enumerate(texts):
#         # 解决输入目标文本的空行导致报错的问题
#         if (len(text.strip()) == 0):
#             continue
#         if (text[-1] not in splits): text += "。" if text_language != "en" else "."
#         print(i18n("实际输入的目标文本(每句):"), text)
#         phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
#         print(i18n("前端处理后的文本(每句):"), norm_text2)
#         if not ref_free:
#             bert = torch.cat([bert1, bert2], 1)
#             all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
#         else:
#             bert = bert2
#             all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
#
#         bert = bert.to(device).unsqueeze(0)
#         all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
#
#         t2 = ttime()
#         # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
#         # print(cache.keys(),if_freeze)
#         '''
#         if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
#         else:
#             with torch.no_grad():
#                 pred_semantic, idx = t2s_model.model.infer_panel(
#                     all_phoneme_ids,
#                     all_phoneme_len,
#                     None if ref_free else prompt,
#                     bert,
#                     # prompt_phone_len=ph_offset,
#                     top_k=top_k,
#                     top_p=top_p,
#                     temperature=temperature,
#                     early_stop_num=hz * max_sec,
#                 )
#                 pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#                 cache[i_text]=pred_semantic
#         '''
#         with torch.no_grad():
#             pred_semantic, idx = t2s_model.model.infer_panel(
#                 all_phoneme_ids,
#                 all_phoneme_len,
#                 None if ref_free else prompt,
#                 bert,
#                 # prompt_phone_len=ph_offset,
#                 top_k=top_k,
#                 top_p=top_p,
#                 temperature=temperature,
#                 early_stop_num=hz * max_sec,
#             )
#             pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#         t3 = ttime()
#         refers=[]
#         '''
#         if(inp_refs):
#             for path in inp_refs:
#                 try:
#                     refer = get_spepc(hps, path.name).to(dtype).to(device)
#                     refers.append(refer)
#                 except:
#                     traceback.print_exc()
#         '''
#         dtype=torch.float16 if is_half == True else torch.float32
#         if(len(refers)==0):refers = [get_spepc(hps, ref_wav).to(dtype).to(device)]
#         audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
#         max_audio=np.abs(audio).max()#简单防止16bit爆音
#         if max_audio>1:audio/=max_audio
#         audio_opt.append(audio)
#         audio_opt.append(zero_wav)
#         t4 = ttime()
#         t.extend([t2 - t1,t3 - t2, t4 - t3])
#         t1 = ttime()
#     print("%.3f\t%.3f\t%.3f\t%.3f" %
#            (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
#            )
#     return torch.Tensor(np.concatenate(audio_opt, 0)).unsqueeze(0)
#     '''
#     yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
#         np.int16
#     )
#     '''


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


resample_transform_dict = {}

def resample(audio_tensor, sr0,sr1):
    global resample_transform_dict
    key="%s-%s"%(sr0,sr1)
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch

spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

bigvgan_model = None
hifigan_model = None

def init_bigvgan():
    global bigvgan_model,hifigan_model
    from GPT_SoVITS.BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if hifigan_model:
        hifigan_model=hifigan_model.cpu()
        hifigan_model=None
        try:torch.cuda.empty_cache()
        except:pass
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

def init_hifigan():
    global hifigan_model,bigvgan_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0, is_bias=True
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load("%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,), map_location="cpu")
    print("loading vocoder",hifigan_model.load_state_dict(state_dict_g))
    if bigvgan_model:
        bigvgan_model=bigvgan_model.cpu()
        bigvgan_model=None
        try:torch.cuda.empty_cache()
        except:pass
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


punctuation = set(['!', '?', '…', ',', '.', '-'," "])

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

class TextDictNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                        "language": (list(dict_language.keys()),)
                    }
                }
    RETURN_TYPES = ("TEXTDICT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_GPT-SoVITS"

    def encode(self,text,language):
        res = {
            "text": text,
            "language":language
        }
        return (res, )


prompt_sr = 32000
import torchaudio
from feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
import librosa

class GSVTTSNode:

    def __init__(self):
        self.GPT_weight = None
        self.SoVITS_weight = None

    @classmethod
    def INPUT_TYPES(s):
        # SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
        return {
            "required":{
                "text_dict": ("TEXTDICT",),
                "prompt_text_dict":("TEXTDICT",),
                "prompt_audio":("AUDIO",),
                "config":("CONFIG",),
                "GPT_weight":("STRING",),
                "SoVITS_weight":("STRING",),
                "how_to_cut":([i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],{
                    "default": i18n("凑四句一切")
                }),
                "speed":("FLOAT",{
                    "min": 0.6,
                    "max":1.65,
                    "step":0.05,
                    "rond": 0.001,
                    "display":"slider",
                    "default": 1.0
                }),
                "top_k":("INT",{
                    "min": 1,
                    "max":100,
                    "step":1,
                    "display":"slider",
                    "default": 15
                }),
                "top_p":("FLOAT",{
                    "min": 0.,
                    "max":1.,
                    "step":0.05,
                    "rond": 0.001,
                    "display":"slider",
                    "default": 1.0
                }),
                "temperature":("FLOAT",{
                    "min": 0.,
                    "max":1.,
                    "step":0.05,
                    "rond": 0.001,
                    "display":"slider",
                    "default": 1.0
                }),
                "pitch":("FLOAT",{
                    "min": -12.,
                    "max":12.,
                    "step":0.5,
                    "display":"slider",
                    "default": 0.0
                }),
                "volume":("FLOAT",{
                    "min": 0.,
                    "max":1.,
                    "step":0.05,
                    "rond": 0.001,
                    "display":"slider",
                    "default": 1.0
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "tts"

    CATEGORY = "AIFSH_GPT-SoVITS"

    def tts(self,text_dict,prompt_text_dict,prompt_audio,
            config,GPT_weight,SoVITS_weight,how_to_cut,
            speed,top_k,top_p,temperature, pitch, volume):
        global ssl_model,is_half,tokenizer,bert_model, version, model_version

        is_half = config['is_half']
        version = "v2"
        model_version = config['version']
        if self.GPT_weight is None:
            cnhubert.cnhubert_base_path = os.path.join(models_dir,"chinese-hubert-base")
            ssl_model = cnhubert.get_model()
            if is_half == True:
                ssl_model = ssl_model.half().to(device)
            else:
                ssl_model = ssl_model.to(device)


            bert_path = os.path.join(models_dir,"chinese-roberta-wwm-ext-large")

            tokenizer = AutoTokenizer.from_pretrained(bert_path)
            bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
            if is_half == True:
                bert_model = bert_model.half().to(device)
            else:
                bert_model = bert_model.to(device)

        text  = text_dict ['text']
        text_language = text_dict['language']
        prompt_text = prompt_text_dict['text']
        prompt_language = prompt_text_dict['language']

        waveform = prompt_audio['waveform'].squeeze(0)
        audio_np = waveform.numpy()
        audio_np = librosa.effects.pitch_shift(audio_np, sr=prompt_sr, n_steps=pitch)
        waveform = torch.from_numpy(audio_np)

        source_sr = prompt_audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        if self.SoVITS_weight != SoVITS_weight:
            self.SoVITS_weight = SoVITS_weight
            # SoVITS_weight_path = os.path.join(models_dir,SoVITS_weight) if "s2G" in SoVITS_weight else os.path.join(work_path,SoVITS_weight)
            SoVITS_weight_path = os.path.join(models_dir,SoVITS_weight + ".pth")
            change_sovits_weights(SoVITS_weight_path, prompt_language=prompt_language, text_language=text_language)

        if self.GPT_weight != GPT_weight:
            self.GPT_weight = GPT_weight
            # GPT_weight_path = os.path.join(models_dir,GPT_weight) if "epoch=" in GPT_weight else os.path.join(work_path,GPT_weight)
            GPT_weight_path = os.path.join(models_dir,GPT_weight + ".ckpt")
            change_gpt_weights(GPT_weight_path)

        res_audio = get_tts_wav(speech.squeeze(0), prompt_text, prompt_language, text,
                                text_language, how_to_cut, top_k, top_p,
                                temperature,speed)

        if model_version == "v4":
            res_audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=prompt_sr)(res_audio)

        res_audio = res_audio * volume
        res = {
            "waveform": res_audio.unsqueeze(0),
            "sample_rate": prompt_sr,
        }
        return (res,)


import srt
import datetime
import traceback
import folder_paths
from tools.slicer2 import Slicer
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
input_dir = folder_paths.get_input_directory()
class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_CosyVoice"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)

class PreViewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }

    CATEGORY = "AIFSH_GPT-SoVITS"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r',encoding="utf-8") as f:
            srt_content = f.read()
        return {"ui": {"srt":[srt_content,srt_name,dir_name]}}

import ffmpeg
def speed_change(input_audio, speed, sr):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

class TSCY_Node:
    def __init__(self):
        self.ifload_model = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "from_language":(list(dict_language.keys()),),
                "to_language": (list(dict_language.keys()),),
                "prompt_audio":("AUDIO",),
                "translator":(['alibaba', 'apertium', 'argos', 'baidu', 'bing',
                            'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
                            'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
                            'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
                            'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
                            'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
                            'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
                            'yeekit', 'youdao'],{
                                "default":"sogou"
                            }),
                "if_algin":("BOOLEAN",{
                    "default": True
                })
            },
            "optional":{
                "tts_srt": ("SRT",),
            }
        }
    RETURN_TYPES = ("AUDIO","SRT",)
    FUNCTION = "tts"

    CATEGORY = "AIFSH_GPT-SoVITS"

    def wishper2gsv(self,lang):
        if lang in "en":
            lang = i18n("英文")
        elif lang in "ko":
            lang = i18n("韩文")
        elif lang in "ja":
            lang = i18n("日文")
        else:
            lang = i18n("中文")
        return lang
    def gsv2translator(self,lang):
        if "en" in lang:
            lang = "en"

        elif "ja" in lang:
            lang = "ja"

        elif "ko" in lang:
            lang = "ko"
        else:
            lang = "zh"
        return lang

    def tts(self,from_language,to_language,prompt_audio,translator,if_algin,tts_srt=None):
        global ssl_model,is_half,tokenizer,bert_model

        waveform = prompt_audio['waveform'].squeeze(0)
        source_sr = prompt_audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)

        if self.ifload_model:
            is_half = True
            cnhubert.cnhubert_base_path = os.path.join(models_dir,"chinese-hubert-base")
            ssl_model = cnhubert.get_model()
            if is_half == True:
                ssl_model = ssl_model.half().to(device)
            else:
                ssl_model = ssl_model.to(device)


            bert_path = os.path.join(models_dir,"chinese-roberta-wwm-ext-large")

            tokenizer = AutoTokenizer.from_pretrained(bert_path)
            bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
            if is_half == True:
                bert_model = bert_model.half().to(device)
            else:
                bert_model = bert_model.to(device)
            change_gpt_weights(os.path.join(models_dir,pretrained_gpt_name[0]))
            change_sovits_weights(os.path.join(models_dir,pretrained_sovits_name[0]))
            self.ifload_model = False
        else:
            print("use cache model")

        slicer = Slicer(
            sr= prompt_sr,
            threshold= -34,
            min_length= 4000,
            min_interval= 300,
            hop_size= 10,
            max_sil_kept= 500
        )
        model_path = os.path.join(models_dir,f"faster-whisper-large-v3")
        snapshot_download(repo_id=f"Systran/faster-whisper-large-v3",local_dir=model_path)
        try:
            model = WhisperModel(model_path, device=device, compute_type="float16")
        except:
            return print(traceback.format_exc())

        tts_audio = []
        tr_to_language = self.gsv2translator(dict_language[to_language])
        whisper_language = self.gsv2translator(dict_language[from_language])
        subs = []
        for i, (chunk, start, end) in enumerate(slicer.slice(speech.numpy()[0])):
            tmp_max = np.abs(chunk).max()
            if(tmp_max>1):chunk/=tmp_max
            chunk = (chunk / tmp_max * (0.9 * 0.25)) + (1 - 0.25) * chunk

            segments, info = model.transcribe(
                audio          = chunk,
                beam_size      = 5,
                vad_filter     = True,
                vad_parameters = dict(min_silence_duration_ms=700),
                language       = whisper_language)
            i_prompt_text = ''
            if i_prompt_text == '':
                for segment in segments:
                    i_prompt_text += segment.text
            i_prompt_audio = torch.from_numpy(chunk)

            print(f"from {whisper_language} \t {i_prompt_text}")
            if tts_srt is None:
                import translators as ts
                i_tts_text = ts.translate_text(query_text=i_prompt_text,from_language=whisper_language,
                                            to_language=tr_to_language,translator=translator)
            else:
                with open(tts_srt,"r",encoding="utf-8") as f:
                    sub_str = f.read()
                i_tts_text = list(srt.parse(sub_str))[i].content

            print(f"to {tr_to_language}\t{i_tts_text}")

            i_sub = srt.Subtitle(index=i+1,start=datetime.timedelta(seconds=start/prompt_sr),
                                    end=datetime.timedelta(seconds=end/prompt_sr),content=i_tts_text)
            subs.append(i_sub)
            i_tts_audio = get_tts_wav(i_prompt_audio,i_prompt_text,from_language,i_tts_text,to_language)
            i_tts_audio = (i_tts_audio.numpy() * 32768).astype(np.int16)
            if if_algin:
                ratio = i_tts_audio.shape[-1] / (end-start)
                i_tts_audio = [speed_change(i_tts_audio,speed=ratio,sr=prompt_sr) / 32768]
                print(f"change speed {ratio}")
            tts_audio.append(i_tts_audio)


        srt_path = os.path.join(output_dir,"tmp.srt")
        res_audio = torch.Tensor(np.concatenate(tts_audio, 0)).unsqueeze(0)
        with open(srt_path,"w",encoding="utf-8") as f:
            f.write(srt.compose(subs))

        res = {
            "waveform": res_audio,
            "sample_rate": prompt_sr,
        }
        return (res,srt_path,)