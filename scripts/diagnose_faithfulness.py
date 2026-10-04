"""Diagnostic only: capture actual Faithfulness statements/verdicts, unchanged judge."""
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['RAGAS_DO_NOT_TRACK']='true'
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig
from config import OPENAI_API_KEY,OPENAI_BASE_URL,EVALUATION_MODEL

class Capture(BaseCallbackHandler):
    def __init__(self):
        self.outputs=[]
    def on_llm_end(self,response,**kwargs):
        self.outputs.append(response.generations[0][0].text)

async def main():
    # This investigation targets v1, whose generator/judge evidence was misaligned.
    original = ROOT/'reports/history/v1/ragas_report.json'
    report=json.loads((original if original.exists() else ROOT/'reports/ragas_report.json').read_text(encoding='utf-8'))
    output=ROOT/'reports/faithfulness_diagnostic.json'
    entries=[]
    llm=ChatOpenAI(model=EVALUATION_MODEL,api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL,temperature=0,timeout=60,max_retries=0)
    metric=Faithfulness(llm=LangchainLLMWrapper(llm))
    metric.init(RunConfig(timeout=120,max_retries=1,max_wait=5))
    for number in (6,7,14):
        row=report['per_question'][number-1]
        sources=report['run_metadata']['retrieval_sources'][number-1]
        variants=[('raw',row['contexts'])]
        if number==6:
            variants.append(('source_labels',[f"[Nguồn: {s['source']}]\n{c}" for s,c in zip(sources,row['contexts'])]))
        for variant,contexts in variants:
            capture=Capture()
            score=await metric.ascore({'question':row['question'],'answer':row['answer'],'contexts':contexts},callbacks=[capture],timeout=120)
            entry={'case_number':number,'variant':variant,'question':row['question'],'answer':row['answer'],
                   'contexts':contexts,'faithfulness':score,'judge_outputs':capture.outputs}
            entries.append(entry)
            output.write_text(json.dumps({'evidence_type':'diagnostic_not_20_case_benchmark','judge_model':EVALUATION_MODEL,'entries':entries},ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'case':number,'variant':variant,'score':score,'judge_outputs':capture.outputs},ensure_ascii=False),flush=True)

asyncio.run(main())
