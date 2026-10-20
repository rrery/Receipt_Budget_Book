"""One-time model cache preparation on a new host; never uses customer receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.pipeline import CACHE,MODEL,LEGACY,ocr_python


def main():
    os.environ.setdefault('HF_HOME',str(CACHE/'huggingface'))
    from transformers import AutoModel,AutoProcessor
    info_path=MODEL/'model_info.json'
    if not info_path.exists() or not (MODEL/'final_model/linear_head.safetensors').exists():
        raise RuntimeError('학습 산출물을 BUDGET_MODEL_DIR에 먼저 배치하세요.')
    info=json.loads(info_path.read_text(encoding='utf-8'))
    AutoProcessor.from_pretrained(info['model_name'],apply_ocr=False)
    backbone=AutoModel.from_pretrained(info['model_name']);del backbone
    environment=os.environ.copy()
    environment['PYTHONUTF8']='1'
    environment['PADDLE_PDX_CACHE_HOME']=os.path.relpath(CACHE/'paddlex',ROOT) if os.name=='nt' else str(CACHE/'paddlex')
    # Import the original class and use precisely the existing Korean OCR settings.
    code="from ysz_ocr import PaddleOCR; PaddleOCR(lang='korean',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,text_det_limit_side_len=1536,text_det_limit_type='max',enable_mkldnn=False)"
    environment['PYTHONPATH']=str(LEGACY)
    subprocess.run([str(ocr_python()),'-c',code],cwd=ROOT,env=environment,check=True)
    print('Model caches ready.',flush=True)


if __name__=='__main__':main()
