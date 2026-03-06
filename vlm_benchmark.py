#!/usr/bin/env python3
"""
vlm_benchmark.py  —  Multi-Model VLM Comparison  [Panoramic Capture Flow]
==========================================================================
Plugs into the existing depth_mapper panoramic capture UI.
Same interface as always: YOLO | SAM | Depth panels, SPACE to capture,
ENTER to finish. After fusion all 4 models run on your captured frames.

Standalone:
    python vlm_benchmark.py

Or add one line to depth_mapper.py after mapper.run() returns:
    from vlm_benchmark import run_vlm_benchmark
    run_vlm_benchmark(frames, accessibility, stats, objects)
"""

import base64, json, re, sys, time, urllib.request
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

MODELS = [
    {'id':'moondream',       'name':'Moondream',  'size':'1.7GB','tier':'Lightweight'},
    {'id':'llava:7b',        'name':'LLaVA-7B',   'size':'4.7GB','tier':'Mid-weight'},
    {'id':'bakllava:latest', 'name':'BakLLaVA',   'size':'4.7GB','tier':'Mid-weight'},
    {'id':'llava:13b',       'name':'LLaVA-13B',  'size':'8.0GB','tier':'Heavyweight'},
]
PROMPTS = [
    {'id':'scene','name':'Scene Description',
     'prompt':'Describe what you see on this bed. List all objects visible.',
     'keywords':['bed','pillow','duvet','cover','mattress','sheet','blanket']},
    {'id':'accessibility','name':'Accessibility',
     'prompt':('Look at this bed. Answer:\n'
               '1. HEAD (top) side blocked? yes/no/partial\n'
               '2. FOOT (bottom) blocked? yes/no/partial\n'
               '3. LEFT blocked? yes/no/partial\n'
               '4. RIGHT blocked? yes/no/partial'),
     'keywords':['head','foot','left','right','blocked','free','yes','no']},
    {'id':'grasp','name':'Grasp Point',
     'prompt':'Where is the pillow? Describe which corner or edge to grab it from.',
     'keywords':['pillow','corner','edge','left','right','top','bottom','grab']},
    {'id':'stepcheck','name':'Step Check',
     'prompt':'Is the bed fully made and neatly arranged? Answer yes or no and explain briefly.',
     'keywords':['yes','no','made','neat','arranged','complete','missing']},
]
GT_QUESTIONS = [
    ('scene',        'Objects ON the bed? (comma-separated e.g. pillow,duvet,bed_cover)'),
    ('head_blocked', 'HEAD side blocked? (yes/no/partial)'),
    ('foot_blocked', 'FOOT side blocked? (yes/no/partial)'),
    ('left_blocked', 'LEFT side blocked? (yes/no/partial)'),
    ('right_blocked','RIGHT side blocked? (yes/no/partial)'),
]
HALLUC = ['desk','laptop','book','chair','office','monitor','keyboard']
OLLAMA = "http://localhost:11434/api/generate"
OUTDIR = Path("results/benchmark")


def _b64(img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''

def _query(mid, prompt, b64, timeout=180):
    payload = json.dumps({'model':mid,'prompt':prompt,'images':[b64],
                          'stream':False,'options':{'temperature':0.1,'num_predict':200}}).encode()
    req = urllib.request.Request(OLLAMA, data=payload,
                                 headers={'Content-Type':'application/json'}, method='POST')
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return data.get('response',''), round(time.time()-t0,2), None
    except Exception as e:
        return '', round(time.time()-t0,2), str(e)

def _auto(response, prompt_cfg):
    r = response.strip().lower()
    kw = prompt_cfg.get('keywords',[])
    hits = sum(1 for k in kw if k in r)
    kw_pct = round(hits/max(len(kw),1)*100,1)
    halluc = sum(1 for t in HALLUC if t in r) >= 2
    ln = len(r)
    ls = 0 if ln<10 else (70 if ln>500 else min(100,ln//2))
    score = (25 if ln>5 else 0) + kw_pct*0.4 + (20 if not halluc else 0) + ls*0.15
    return {'non_empty':ln>5,'keyword_pct':kw_pct,'keyword_hits':hits,
            'hallucinated':halluc,'halluc_terms':[t for t in HALLUC if t in r],
            'length':ln,'auto_score':round(score,1)}

def _gt_score(response, gt, pid):
    r = response.strip().lower()
    if pid == 'accessibility':
        mapping = {s:gt.get(f'{s}_blocked','') for s in ('head','foot','left','right')}
        correct = sum(1 for side,exp in mapping.items()
                      if exp and side in r and exp[:3] in r[max(0,r.find(side)-5):r.find(side)+60])
        total = sum(1 for v in mapping.values() if v)
        return {'gt_score':round(correct/max(total,1)*100,1)}
    elif pid == 'scene':
        objs = [o.strip() for o in gt.get('scene','').split(',') if o.strip()]
        found = [o for o in objs if o in r]
        return {'gt_score':round(len(found)/max(len(objs),1)*100,1),'found':found,
                'missed':[o for o in objs if o not in found]}
    return {'gt_score':None}

def _collect_gt(frames_rgb, frame_names):
    print(f"\n{'─'*55}\n  GROUND TRUTH LABELLING\n{'─'*55}")
    print("  Answer 5 questions per frame. A window shows each frame.\n")
    gt = {}
    cv2.namedWindow("GT Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GT Frame", 640, 480)
    for i,(rgb,name) in enumerate(zip(frames_rgb,frame_names)):
        print(f"  Frame {i+1}/{len(frames_rgb)}: {name}")
        cv2.imshow("GT Frame", rgb); cv2.waitKey(400)
        answers = {}
        for key,question in GT_QUESTIONS:
            while True:
                ans = input(f"    {question}: ").strip().lower()
                if ans: answers[key]=ans; break
        gt[name] = answers; print()
    cv2.destroyWindow("GT Frame")
    return gt

def _progress(model_name, prompt_name, done, total, win):
    W,H = 1280,420
    p = np.full((H,W,3),12,dtype=np.uint8)
    cv2.putText(p,"VLM BENCHMARK IN PROGRESS",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,220,255),2,cv2.LINE_AA)
    bw = W-60; filled = int(bw*done/max(total,1))
    cv2.rectangle(p,(30,68),(30+bw,90),(25,35,45),-1)
    cv2.rectangle(p,(30,68),(30+filled,90),(0,200,100),-1)
    cv2.putText(p,f"{done}/{total}  ({done*100//max(total,1)}%)",(30,115),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,(120,120,120),1,cv2.LINE_AA)
    cv2.putText(p,f"Model : {model_name}",(30,155),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(p,f"Prompt: {prompt_name}",(30,188),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1,cv2.LINE_AA)
    tc = {'Lightweight':(0,220,255),'Mid-weight':(100,255,100),'Heavyweight':(255,140,0)}
    for j,m in enumerate(MODELS):
        col = tc.get(m['tier'],(140,140,140))
        prefix = ">> " if m['name']==model_name else "   "
        cv2.putText(p,f"{prefix}{m['name']}  ({m['size']}, {m['tier']})",
                    (30,255+j*33),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)
    cv2.putText(p,"Window updates automatically — do not close.",
                (30,H-18),cv2.FONT_HERSHEY_SIMPLEX,0.40,(60,60,60),1,cv2.LINE_AA)
    cv2.imshow(win, p); cv2.waitKey(1)

def _run_benchmark(frames_rgb, frame_names, gt, win):
    total = len(MODELS)*len(frames_rgb)*len(PROMPTS)
    done  = 0
    print(f"\n{'─'*55}")
    print(f"  {len(MODELS)} models x {len(frames_rgb)} frames x {len(PROMPTS)} prompts = {total} calls")
    print(f"  Est. time: {total*55//60}–{total*90//60} min  (llava:13b is slowest)")
    print(f"{'─'*55}\n")
    b64s = [_b64(rgb) for rgb in frames_rgb]
    per_response = []
    for model in MODELS:
        mid,mname = model['id'],model['name']
        print(f"  ── {mname} ({model['size']}) ──")
        for fi,(b64,fname) in enumerate(zip(b64s,frame_names)):
            for pc in PROMPTS:
                done += 1
                _progress(mname, pc['name'], done-1, total, win)
                print(f"    [{done:3d}/{total}] {fname} / {pc['name']}…",end='',flush=True)
                resp,elapsed,err = _query(mid, pc['prompt'], b64)
                a = _auto(resp, pc)
                entry = {'model_id':mid,'model_name':mname,'frame':fname,
                         'prompt_id':pc['id'],'prompt_name':pc['name'],
                         'response':resp,'elapsed':elapsed,'error':err,'auto':a,'gt':{}}
                if gt and fname in gt:
                    entry['gt'] = _gt_score(resp, gt[fname], pc['id'])
                per_response.append(entry)
                st = '✓' if a['non_empty'] and not a['hallucinated'] else '⚠'
                print(f" {st} {elapsed:.1f}s  kw={a['keyword_pct']}%  score={a['auto_score']}")
        print()
    summary = {}
    for model in MODELS:
        mid = model['id']
        rows  = [r for r in per_response if r['model_id']==mid]
        valid = [r for r in rows if r['auto']['non_empty']]
        times = [r['elapsed'] for r in rows]
        gt_sc = [r['gt']['gt_score'] for r in rows
                 if r.get('gt') and r['gt'].get('gt_score') is not None]
        summary[mid] = {
            'name':model['name'],'tier':model['tier'],'size':model['size'],
            'total_calls':len(rows),'valid_responses':len(valid),
            'parse_rate':round(len(valid)/max(len(rows),1)*100,1),
            'halluc_count':sum(1 for r in rows if r['auto']['hallucinated']),
            'avg_time':round(sum(times)/max(len(times),1),2),
            'avg_keyword_pct':round(sum(r['auto']['keyword_pct'] for r in rows)/max(len(rows),1),1),
            'avg_auto_score':round(sum(r['auto']['auto_score'] for r in rows)/max(len(rows),1),1),
            'avg_gt_score':round(sum(gt_sc)/len(gt_sc),1) if gt_sc else None,
        }
    return {'timestamp':datetime.now().isoformat(),'models':MODELS,'prompts':PROMPTS,
            'frames':frame_names,'has_gt':bool(gt),'per_response':per_response,'summary':summary}

def _report(results, frames_rgb, accessibility, stats, out_path):
    S=results['summary']; PR=results['per_response']
    has_gt=results['has_gt']; ts=results['timestamp'][:19].replace('T',' ')
    mids=[m['id'] for m in MODELS]; mnames={m['id']:m['name'] for m in MODELS}
    fnames=results['frames']; pids=[p['id'] for p in PROMPTS]
    pnames={p['id']:p['name'] for p in PROMPTS}

    # Embed frame thumbnails
    thumbs={}
    for name,rgb in zip(fnames,frames_rgb):
        sm=cv2.resize(rgb,(200,150))
        ok,buf=cv2.imencode('.jpg',sm,[cv2.IMWRITE_JPEG_QUALITY,72])
        if ok: thumbs[name]=base64.b64encode(buf.tobytes()).decode()

    icons={'free':'✓','partially_blocked':'~','blocked':'✗','unknown':'?'}
    acc_html=''.join(
        f'<span class="a{v}">{s.upper()}: {icons.get(v,"?")} {v.upper()}'
        f'{" · "+str(round(stats.get(s,{}).get("median_depth_m",0),2))+"m" if s in stats else ""}'
        f'</span><br>' for s,v in accessibility.items())

    def cell(mid,fname,pid):
        row=next((r for r in PR if r['model_id']==mid and r['frame']==fname and r['prompt_id']==pid),None)
        if not row: return '<td class="empty">—</td>'
        a=row['auto']; resp=row['response'].strip()
        disp=resp[:260]+('…' if len(resp)>260 else '') if resp else '<em class="nr">no response</em>'
        sc=a['auto_score']; cls='g' if sc>=60 else ('w' if sc>=30 else 'b')
        gtb=''
        if has_gt and row.get('gt') and row['gt'].get('gt_score') is not None:
            gtb=f'<span class="gtb">GT {row["gt"]["gt_score"]:.0f}%</span>'
        hb='<span class="hb">HALLUC</span>' if a['hallucinated'] else ''
        return (f'<td class="rc {cls}"><div class="rm">'
                f'<span class="sb">{sc:.0f}</span><span class="tb">{row["elapsed"]:.1f}s</span>'
                f'<span class="kb">kw {a["keyword_pct"]:.0f}%</span>{gtb}{hb}</div>'
                f'<div class="rt">{disp}</div></td>')

    fsections=''
    for fname in fnames:
        th=f'<img src="data:image/jpeg;base64,{thumbs[fname]}" class="th">' if fname in thumbs else ''
        ptables=''
        for pid in pids:
            hdrs=''.join(f'<th>{mnames[mid]}</th>' for mid in mids)
            cells=''.join(cell(mid,fname,pid) for mid in mids)
            ptables+=f'<div class="pb"><div class="pl">{pnames[pid]}</div><div class="rs"><table class="rt2"><thead><tr>{hdrs}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div></div>'
        fsections+=f'<details class="fd" open><summary class="fs">{th}<span class="fn">{fname}</span></summary>{ptables}</details>'

    srows=''
    for m in MODELS:
        mid=m['id']
        if mid not in S: continue
        s=S[mid]; gt=f"{s['avg_gt_score']:.1f}%" if s['avg_gt_score'] is not None else '—'
        tc={'Lightweight':'#00e5ff','Mid-weight':'#4ade80','Heavyweight':'#fb923c'}.get(s['tier'],'#aaa')
        srows+=f'<tr><td><strong>{s["name"]}</strong><br><span class="tt" style="color:{tc}">{s["tier"]} · {s["size"]}</span></td><td class="n">{s["avg_auto_score"]:.1f}</td><td class="n">{s["avg_keyword_pct"]:.1f}%</td><td class="n">{s["parse_rate"]:.1f}%</td><td class="n">{s["halluc_count"]}</td><td class="n">{s["avg_time"]:.1f}s</td><td class="n gc">{gt}</td></tr>'

    cL=json.dumps([S[m]['name'] for m in mids if m in S])
    cS=json.dumps([S[m]['avg_auto_score'] for m in mids if m in S])
    cK=json.dumps([S[m]['avg_keyword_pct'] for m in mids if m in S])
    cT=json.dumps([S[m]['avg_time'] for m in mids if m in S])
    cP=json.dumps([S[m]['parse_rate'] for m in mids if m in S])
    cG=json.dumps([S[m]['avg_gt_score'] or 0 for m in mids if m in S])

    html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM Benchmark</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600&display=swap');
:root{{--bg:#07090d;--s:#0d1117;--b:#161b22;--a:#58a6ff;--g:#3fb950;--o:#d29922;--r:#f85149;--t:#c9d1d9;--m:#8b949e}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--t);font-size:14px;line-height:1.6}}
h1{{font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:var(--a)}}
h2{{font-family:'JetBrains Mono',monospace;font-size:.72rem;text-transform:uppercase;letter-spacing:3px;color:var(--m);margin-bottom:1rem}}
header{{padding:1.4rem 2rem;border-bottom:1px solid var(--b);display:flex;justify-content:space-between;align-items:center;gap:1.5rem;flex-wrap:wrap}}
.hacc{{font-family:'JetBrains Mono',monospace;font-size:.68rem;line-height:2}}
.afree{{color:var(--g)}}.ablocked{{color:var(--r)}}.apartially_blocked{{color:var(--o)}}.aunknown{{color:var(--m)}}
.con{{max-width:1700px;margin:0 auto;padding:1.5rem 2rem}}
.sec{{margin-bottom:2.5rem}}
.rg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem}}
.rc2{{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:1rem;position:relative;overflow:hidden}}
.rc2::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
.r1::after{{background:var(--a)}}.r2::after{{background:var(--g)}}.r3::after{{background:var(--o)}}.r4::after{{background:var(--m)}}
.rnum{{font-family:'JetBrains Mono',monospace;font-size:2.2rem;color:var(--b);position:absolute;right:.6rem;top:.2rem;font-weight:600}}
.rname{{font-weight:600;margin-bottom:.2rem}}
.rscore{{font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:var(--a)}}
.rdet{{font-size:.72rem;color:var(--m);margin-top:.4rem;line-height:1.9}}
.st{{width:100%;border-collapse:collapse}}
.st th{{text-align:left;padding:.45rem .9rem;font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--m);border-bottom:1px solid var(--b)}}
.st td{{padding:.65rem .9rem;border-bottom:1px solid var(--b)}}
.st tr:hover td{{background:rgba(255,255,255,.015)}}
.n{{font-family:'JetBrains Mono',monospace;font-size:.8rem}}.gc{{color:var(--a)}}
.tt{{font-size:.65rem;font-family:'JetBrains Mono',monospace}}
.cg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}}
.cc{{background:var(--s);border:1px solid var(--b);border-radius:7px;padding:.9rem}}
.ctl{{font-family:'JetBrains Mono',monospace;font-size:.65rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--m);margin-bottom:.7rem}}
.fd{{background:var(--s);border:1px solid var(--b);border-radius:7px;margin-bottom:.8rem;overflow:hidden}}
.fs{{display:flex;align-items:center;gap:.8rem;padding:.7rem 1rem;cursor:pointer;list-style:none;user-select:none}}
.fs::-webkit-details-marker{{display:none}}.fs:hover{{background:rgba(255,255,255,.015)}}
.th{{width:72px;height:54px;object-fit:cover;border-radius:3px;border:1px solid var(--b)}}
.fn{{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--a)}}
.pb{{padding:.7rem 1rem .9rem;border-top:1px solid var(--b)}}
.pl{{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--m);text-transform:uppercase;letter-spacing:1px;margin-bottom:.45rem}}
.rs{{overflow-x:auto}}
.rt2{{width:100%;border-collapse:collapse;min-width:860px}}
.rt2 th{{font-family:'JetBrains Mono',monospace;font-size:.66rem;text-transform:uppercase;letter-spacing:1px;color:var(--a);padding:.35rem .7rem;border-bottom:1px solid var(--b);text-align:left;min-width:240px}}
.rc{{vertical-align:top;padding:.55rem .7rem;border-right:1px solid var(--b)}}
.rc.g{{background:rgba(63,185,80,.05)}}.rc.w{{background:rgba(210,153,34,.05)}}.rc.b{{background:rgba(248,81,73,.05)}}
.rm{{display:flex;flex-wrap:wrap;gap:.25rem;margin-bottom:.35rem}}
.sb{{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(88,166,255,.12);color:var(--a);padding:1px 5px;border-radius:3px}}
.tb{{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(139,148,158,.12);color:var(--m);padding:1px 5px;border-radius:3px}}
.kb{{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(63,185,80,.1);color:var(--g);padding:1px 5px;border-radius:3px}}
.gtb{{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(88,166,255,.15);color:var(--a);padding:1px 5px;border-radius:3px}}
.hb{{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(248,81,73,.18);color:var(--r);padding:1px 5px;border-radius:3px}}
.rt{{font-size:.78rem;color:var(--t);line-height:1.5;white-space:pre-wrap;max-height:100px;overflow-y:auto}}
.nr{{color:var(--r);font-style:italic}}.empty{{color:var(--m);text-align:center}}
</style></head><body>
<header>
<div><h1>VLM Benchmark</h1>
<p style="color:var(--m);font-family:'JetBrains Mono',monospace;font-size:.68rem;margin-top:.2rem">Bed-Making Robot · {ts} · {len(fnames)} frames · {len(PROMPTS)} prompts · {len(MODELS)} models</p></div>
<div class="hacc"><strong style="font-size:.65rem;color:var(--m)">ACCESSIBILITY</strong><br>{acc_html}</div>
<div style="color:var(--m);font-family:'JetBrains Mono',monospace;font-size:.68rem;text-align:right">{"✓ Ground truth enabled" if has_gt else "⚠ Automated metrics only"}<br>{len(MODELS)*len(fnames)*len(PROMPTS)} total calls</div>
</header>
<div class="con">
<div class="sec"><h2>Rankings</h2><div class="rg" id="rg"></div></div>
<div class="sec"><h2>Summary</h2><div style="overflow-x:auto"><table class="st"><thead><tr><th>Model</th><th>Auto Score</th><th>Keyword %</th><th>Parse Rate</th><th>Hallucinations</th><th>Avg Time</th><th class="gc">GT Accuracy</th></tr></thead><tbody>{srows}</tbody></table></div></div>
<div class="sec"><h2>Charts</h2><div class="cg">
<div class="cc"><div class="ctl">Auto Score ↑</div><canvas id="c1"></canvas></div>
<div class="cc"><div class="ctl">Keyword Coverage % ↑</div><canvas id="c2"></canvas></div>
<div class="cc"><div class="ctl">Avg Time (s) ↓</div><canvas id="c3"></canvas></div>
<div class="cc"><div class="ctl">Parse Rate % ↑</div><canvas id="c4"></canvas></div>
{"<div class='cc'><div class='ctl'>Ground Truth Accuracy % ↑</div><canvas id='c5'></canvas></div>" if has_gt else ""}
</div></div>
<div class="sec"><h2>Response Details</h2>{fsections}</div>
</div>
<script>
const L={cL},S={cS},K={cK},T={cT},P={cP},G={cG},SM={json.dumps(S)};
const C=['#58a6ff','#3fb950','#d29922','#a371f7'];
const bo={{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#8b949e',font:{{family:'JetBrains Mono',size:9}}}},grid:{{color:'#161b22'}}}},y:{{ticks:{{color:'#8b949e',font:{{family:'JetBrains Mono',size:9}}}},grid:{{color:'#161b22'}}}}}}}};
const bar=d=>{{return{{data:d,backgroundColor:C.map(c=>c+'1a'),borderColor:C,borderWidth:2,borderRadius:3}}}};
['c1','c2','c3','c4'].forEach((id,i)=>{{
  new Chart(document.getElementById(id),{{type:'bar',data:{{labels:L,datasets:[bar([S,K,T,P][i])]}},options:{{...bo}}}});
}});
{"new Chart(document.getElementById('c5'),{type:'bar',data:{labels:L,datasets:[bar(G)]},options:{...bo}});" if has_gt else ""}
const ranked=Object.entries(SM).map(([id,s])=>{{return{{id,...s}}}})
  .sort((a,b)=>(b.avg_auto_score+(b.avg_gt_score||0))-(a.avg_auto_score+(a.avg_gt_score||0)));
const rg=document.getElementById('rg');
ranked.forEach((m,i)=>{{
  const gt=m.avg_gt_score!=null?`<br>GT: <strong>${{m.avg_gt_score}}%</strong>`:'';
  rg.innerHTML+=`<div class="rc2 r${{i+1}}"><span class="rnum">#${{i+1}}</span>
    <div class="rname">${{m.name}}</div>
    <div class="rscore">${{m.avg_auto_score}}<small style="font-size:.7rem;color:#8b949e">/100</small></div>
    <div class="rdet">${{m.tier}} · ${{m.size}}<br>Parse: <strong>${{m.parse_rate}}%</strong><br>Time: <strong>${{m.avg_time}}s</strong><br>Halluc: <strong>${{m.halluc_count}}</strong>${{gt}}</div>
  </div>`;
}});
</script></body></html>"""
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(html)
    print(f"  ✓ Report → {out_path}")
    return out_path


def run_vlm_benchmark(captured_frames, accessibility=None, stats=None,
                      objects=None, win_name="Panoramic Capture"):
    if not captured_frames:
        print("[Benchmark] No frames — skipping"); return
    accessibility = accessibility or {}
    stats         = stats or {}
    frames_rgb    = [f.rgb for f in captured_frames]
    frame_names   = [f"frame_{f.index+1:02d}.png" for f in captured_frames]

    ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTDIR / ts_str
    run_dir.mkdir(parents=True, exist_ok=True)
    for name,rgb in zip(frame_names,frames_rgb):
        cv2.imwrite(str(run_dir/name), rgb)
    print(f"\n[Benchmark] {len(frames_rgb)} frames saved → {run_dir}/")

    print("\n[Benchmark] Collect ground truth labels? (y/N): ", end='', flush=True)
    try:    ans = input().strip().lower()
    except: ans = 'n'
    gt = _collect_gt(frames_rgb, frame_names) if ans == 'y' else {}

    results = _run_benchmark(frames_rgb, frame_names, gt, win_name)
    rjson   = run_dir / "results.json"
    rjson.write_text(json.dumps(results, indent=2))
    print(f"[Benchmark] JSON → {rjson}")

    rhtml = run_dir / "report.html"
    _report(results, frames_rgb, accessibility, stats, rhtml)

    print(f"\n{'═'*55}\n  BENCHMARK COMPLETE\n{'═'*55}")
    ranked = sorted(results['summary'].values(),
                    key=lambda s: s['avg_auto_score'], reverse=True)
    for i,s in enumerate(ranked):
        gt_s = f"  GT={s['avg_gt_score']}%" if s['avg_gt_score'] is not None else ""
        print(f"  #{i+1} {s['name']:12s}  score={s['avg_auto_score']:5.1f}  "
              f"time={s['avg_time']:5.1f}s  parse={s['parse_rate']:5.1f}%  "
              f"halluc={s['halluc_count']}{gt_s}")
    print(f"\n  xdg-open {rhtml}\n{'═'*55}")

    try:
        import subprocess; subprocess.Popen(['xdg-open', str(rhtml)])
    except: pass

    W,H = 1280,200
    p = np.full((H,W,3),12,dtype=np.uint8)
    cv2.putText(p,"BENCHMARK COMPLETE — report opened in browser",
                (30,60),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,220,255),2,cv2.LINE_AA)
    cv2.putText(p,str(rhtml),(30,105),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1,cv2.LINE_AA)
    cv2.putText(p,"Press any key to close",(30,150),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,80,80),1,cv2.LINE_AA)
    cv2.imshow(win_name, p); cv2.waitKey(0)
    return results


def main():
    print("\n"+"="*55+"\n  VLM BENCHMARK — STANDALONE\n"+"="*55)
    sys.path.insert(0, str(Path(__file__).parent))
    from depth_mapper import InteractivePanoramicMapper
    cv2.namedWindow("Panoramic Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Panoramic Capture", 1920, 600)
    cv2.waitKey(1)
    mapper = InteractivePanoramicMapper()
    try:
        acc, stats, objects, frames = mapper.run()
        run_vlm_benchmark(frames, acc, stats, objects, win_name="Panoramic Capture")
    finally:
        mapper.cleanup()

if __name__ == "__main__":
    main()
