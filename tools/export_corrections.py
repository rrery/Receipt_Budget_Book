"""Export consented field corrections as annotation candidates, never fake BIO labels."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.store import Store


def export_candidates(directory: Path,output: Path):
    store=Store(directory)
    if output.exists() and any(output.iterdir()):raise ValueError('출력 폴더가 비어 있지 않습니다.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    with store.connect() as db:
        rows=db.execute("SELECT c.*,r.image_path FROM corrections c JOIN receipts r ON r.id=c.receipt_id WHERE c.training_allowed=1 AND r.allow_training=1 AND r.image_path IS NOT NULL AND c.ocr_snapshot IS NOT NULL ORDER BY c.created_at").fetchall()
    count=0
    with (output/'candidates.jsonl').open('w',encoding='utf-8') as file:
        for row in rows:
            image=Path(row['image_path']).resolve()
            if not image.is_relative_to((directory/'uploads').resolve()) or not image.is_file():continue
            name=row['receipt_id']+'.png'
            shutil.copy2(image,output/'images'/name)
            value={
                'correction_id':row['id'],'receipt_id':row['receipt_id'],'image_path':'images/'+name,
                'image_sha256':row['content_hash'],'before':json.loads(row['before_json']),
                'after':json.loads(row['after_json']),'changed_fields':json.loads(row['changed_fields']),
                'ocr_snapshot':json.loads(row['ocr_snapshot']),
                'model_predictions':json.loads(row['predictions_snapshot']) if row['predictions_snapshot'] else None,
                'engine':row['engine'],'created_at':row['created_at'],
                'needs_annotation':True,'bio_labels_ready':False,
            }
            file.write(json.dumps(value,ensure_ascii=False)+'\n');count+=1
    (output/'manifest.json').write_text(json.dumps({'candidate_count':count,'format':'field-correction-candidates-v1','requires_token_alignment_review':True,'memos_included':False},indent=2),encoding='utf-8')
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(os.getenv('BUDGET_DATA_DIR',ROOT/'runtime')))
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/training_candidates'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    print('Exported candidates:',export_candidates(args.data_dir,args.output))
