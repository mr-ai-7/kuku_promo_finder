"""Optional model-swap layer: use any OpenAI-compatible chat completion API.

This is an enhancement, not required for offline execution. It reranks *existing,
validated* time segments and never lets the model fabricate timestamps. If no
API key is supplied, the offline NLP tool remains fully functional.
"""
from __future__ import annotations
import json
from pathlib import Path
from promo_finder import Episode, selected_dialogue, ts_to_s, s_to_ts, relevant_events


def rerank_with_llm(promos:list[dict], episodes:list[Episode], api_key:str, model:str='gpt-4.1-mini',base_url:str|None=None):
    if not api_key:raise ValueError('API key required for optional model reranking.')
    from openai import OpenAI
    client=OpenAI(api_key=api_key,**({'base_url':base_url} if base_url else {}))
    episodes_by_number={e.number:e for e in episodes}
    reviewed=[]
    for candidate in promos:
        sg=candidate['segments'][0];episode=episodes_by_number[sg['episode']]
        start,end=ts_to_s(sg['start']),ts_to_s(sg['end'])
        lines=selected_dialogue(episode,start,end)
        items=[{'speaker':e.label,'time':s_to_ts(e.start),'line':e.text} for e in lines]
        # Keep the context local, without giving copyrighted episode files to an
        # external public repository; API transfer requires user consent.
        prompt={
           'episode':candidate['episode'], 'start':sg['start'],'end':sg['end'],
           'dialogue':items,
        }
        sys_msg=("You are an acquisition editor evaluating a 20-60 second microdrama promo. "
           "Output ONLY valid JSON with fields hook_0_to_5, cliffhanger_0_to_5, "
           "standalone_0_to_5, verdict ('strong'|'needs_review'|'reject'), "
           "reason (one concise sentence), weakness (one concise sentence). "
           "Judge the first 3 seconds and the final cut, do not invent unseen footage, "
           "do not change timestamps, do not claim prediction of conversion.")
        try:
            completion=client.chat.completions.create(
                model=model,temperature=0,response_format={'type':'json_object'},
                messages=[{'role':'system','content':sys_msg},{'role':'user','content':json.dumps(prompt,ensure_ascii=False)}]
            )
            obj=json.loads(completion.choices[0].message.content)
            for k in ('hook_0_to_5','cliffhanger_0_to_5','standalone_0_to_5'):
                obj[k]=max(0,min(5,float(obj[k])))
            combined=(obj['hook_0_to_5']*.4+obj['cliffhanger_0_to_5']*.4+obj['standalone_0_to_5']*.2)
            item=dict(candidate)
            item['llm_review']=obj
            item['llm_score_out_of_5']=round(combined,3)
            reviewed.append(item)
        except Exception as exc:
            # Do not pretend the model reviewed anything after an API failure.
            raise RuntimeError(f'Model rerank failed for {candidate.get("promo_id")} ({type(exc).__name__}): {exc}') from exc
    reviewed.sort(key=lambda c:-c['llm_score_out_of_5'])
    for i,item in enumerate(reviewed,1):item['llm_rank']=i
    return reviewed
