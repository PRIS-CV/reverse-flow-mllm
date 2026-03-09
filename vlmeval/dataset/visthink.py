import os
import json
import pandas as pd
import re
from .image_mcq import ImageMCQDataset
from ..utils import  track_progress_rich
from .utils import build_judge, DEBUG_MESSAGE
from .utils.multiple_choice import report_acc
from ..smp import *
from ..smp.file import get_intermediate_file_path
from PIL import Image
from .image_base import img_root_map
def toliststr(s):
    import math
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError


class VisThinkBench(ImageMCQDataset):
    TYPE = "MCQ"
    INTERLEAVE = True  # 明确告知是 Interleave 图文输入
    DATASET_URL = {}

    DATASET_MD5 = {}
  

    
    @classmethod
    def supported_datasets(cls):
        return ["VisThink"]
    

    def evaluate(self, eval_file, **judge_kwargs):
        """根据题型计算准确率"""
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        
        nproc = judge_kwargs.pop('nproc', 4)

        model = judge_kwargs.get('model', 'extract_matching')
        # assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
       
        if model == 'exact_matching':
            model = None
        
        elif gpt_key_set():
            if model is None:
                model = 'chatgpt-0125'
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)
        
        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, get_intermediate_file_path(eval_file, f'_{name_str}_result'))
        data = load(get_intermediate_file_path(eval_file, f'_{name_str}_result'))

        acc = report_acc(data)

        dump(acc, score_file)

        return acc


def mcq_vanilla_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}


   
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []
    # from IPython import embed; embed()
    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
    keys = [x['index'] for x in items]
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
  
    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]
    if 'GT' in data:
        data.pop('GT')
    return data 

def eval_vanilla(model, item, dataset_name=None):
    
    
    if item['category'] == "Collision Detection" or item['category'] == "3D Occlusion Judgment":
        retry = 3
        prompt = build_prompt_judge(item['question'], item['prediction'],item['answer'],)
        while retry:
            retry=retry-1
            ans = model.generate(prompt)
            # print("模型输出:", ans)
            # 正则表达式匹配 <result> 中的内容
            pattern_result = r'<result>(.*?)</result>'

            # 正则表达式匹配 <think> 中的内容
            pattern_think = r'<think>(.*?)</think>'

            # 使用 re.search 查找 <result> 内容
            match_result = re.search(pattern_result, ans, re.DOTALL)
            if match_result:
                result_content = match_result.group(1).strip()  # 提取 <result> 中的内容，并去除多余的空格
                print("结果:", result_content)
            else:
                print("没有匹配到 <result> 内容")
                result_content =0

            # 使用 re.search 查找 <think> 内容
            match_think = re.search(pattern_think, ans, re.DOTALL)
            if match_think:
                think_content = match_think.group(1).strip()  # 提取 <think> 中的内容，并去除多余的空格
                print("原因:", think_content)
            else:
                print("没有匹配到 <think> 内容")
                think_content="none"
                
            return dict(hit=int(result_content), log=think_content)
    
    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    opt, match_log = res['opt'], res['log']
    if can_infer(opt) == item['GT']:
        return dict(hit=1, log=f'Match Log: {match_log}. ')
    else:
        return dict(hit=0, log=f'Match Log: {match_log}. ')
    
    


def extract_answer_from_item( model, item, dataset_name=None):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)



    
    prompt = build_prompt_GPT(item['question'], item['prediction'])
    retry = 3


    ret = can_infer(item['prediction'])
    if ret:
        return dict(opt=ret, log=item['prediction'])
    
    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
        else:
            ret = can_infer(ans)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(
                    f'Failed to in infer: prediction is {ans},'
                    f', Answer is {item["answer"]}' if "answer" in item else ""
                )
        retry -= 1

        if retry == 0:
            
            return dict(opt='A', log='Failed to predict, thus randomly generate one. ')

def build_prompt_GPT(question, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of question. '
        'You are provided with a question, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options) or yes or no, and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: Is the right grey box smaller than the left grey box? Please answer with YES or NO.\n'
        'Answer: Yes, the right grey box is smaller than the left grey box.\nYour output: Yes\n'
        'Example 3: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 4: \n'
        'Question: {}?\n\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, prediction)



def build_prompt_judge(question, prediction, answer):
    tmpl = f"""You will act as a strict, impartial evaluator (Judge).
Your task is to evaluate whether the [Prediction] correctly answers the [Input Question], based on the provided [Ground Truth] (standard answer).

**Evaluation Criteria:**

* **1 (Correct):** The [Prediction] is factually correct and aligns with the core meaning of the [Ground Truth]. Trivial wording differences are acceptable, but all key information points must be included and accurate.
* **0 (Incorrect):** The [Prediction] contains any factual errors, omits key information from the [Ground Truth], adds irrelevant or incorrect information, or contradicts the meaning of the [Ground Truth].

**Evaluation Task:**
    [Input Question]: {question}
    [Ground Truth]: {answer}
    [Prediction]: {prediction}
**Output:**

Provide your judgment and reasoning strictly in the following format, with no additional text or explanations.

<result>
[Enter 1 or 0 here]
</result>

<think>
briefly explaining why the [Prediction] was judged as 1 or 0]
</think>
"""
    return tmpl

def can_infer(answer):
        """
        一个简化的函数，用于从模型的回复中推断出选择项。
        
        它会按顺序尝试：
        1. 查找 "拒绝回答" 的关键词 (返回 'Z')。
        2. 查找唯一的单个大写字母选项 (A-Z)，同时会尝试避免 "A cat..." 这样的冠词。
        3. 查找唯一的 "yes" 或 "no" (不区分大小写)。
        
        如果都找不到，或者存在歧义 (例如 "A and B" 或 "yes and no")，则返回 False。
        """
        if not isinstance(answer, str):
            answer = str(answer)

        # 1. 检查是否为 "拒绝回答"
        reject_to_answer = [
            "Sorry, I can't help with images of people yet.",
            "I can't process this file.",
            "I'm sorry, but without the image provided",
            'Cannot determine the answer',
            'Failed to obtain answer via API' # 来自旧的 can_infer
        ]
        for err in reject_to_answer:
            if err in answer:
                return 'Z'

        # 准备用于清理的标点符号
        # 我们需要两次清理：一次区分大小写 (A-Z)，一次不区分 (yes/no)
        chars_to_replace = '.()[],:;!*#{}""\''

        # ---
        # 2. 检查大写字母选项 (A-Z)
        # ---
        # 这一步必须区分大小写，所以不用 .lower()
        answer_mod = answer
        for c in chars_to_replace:
            answer_mod = answer_mod.replace(c, ' ')
        
        # 分割并去除空字符串
        splits = [x.strip() for x in answer_mod.split() if x.strip()]
        
        choices_keys = set(string.ascii_uppercase)
        found_keys = [word for word in splits if word in choices_keys]

        if len(found_keys) == 1:
            ch = found_keys[0]
            
            # 尝试处理 'A' 作为冠词的情况 (例如 "A cat...")
            # 如果 'A' 是第一个词，并且后面还有其他词，则很有可能是冠词，忽略它。
            if ch == 'A' and splits.index('A') == 0 and len(splits) > 1:
                pass # 是冠词，不返回 'A'，让逻辑继续到 yes/no 检查
            else:
                # 找到了唯一的选项 (例如 "The answer is C." 或 "C")
                return ch
        
        # 如果没找到 A-Y，但找到了 'Z'
        if not found_keys and 'Z' in splits:
            return 'Z'

        # ---
        # 3. 检查 "yes" / "no"
        # ---
        # 这一步不区分大小写
        answer_low = answer.lower()
        for c in chars_to_replace:
            answer_low = answer_low.replace(c, ' ')
            
        splits_low = [x.strip() for x in answer_low.split() if x.strip()]

        has_yes = 'yes' in splits_low
        has_no = 'no' in splits_low

        # 只有 "yes" 没有 "no"
        if has_yes and not has_no:
            return 'Yes'
        
        # 只有 "no" 没有 "yes"
        if has_no and not has_yes:
            return 'No'
        
        # 如果两个都有 (例如 "yes or no") 或两个都没有，我们无法判断

        # ---
        # 4. 最终回退
        # ---
        # 如果所有检查都失败了
        return False