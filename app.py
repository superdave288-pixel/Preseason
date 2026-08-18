import math
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

try:
    import nflreadpy as nfl
except Exception:
    nfl = None

st.set_page_config(page_title='NFL Prop Probability', page_icon='🏈', layout='centered')
st.title('🏈 NFL Prop Probability App')
st.caption('Anytime TD + rushing, receiving, and passing yards with preseason support.')

def american_to_prob(o):
    o=float(o); return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def prob_to_american(p):
    p=min(max(float(p),.0001),.9999)
    return int(round(-100*p/(1-p))) if p>=.5 else int(round(100*(1-p)/p))

def to_pd(x):
    return x.to_pandas() if hasattr(x,'to_pandas') else pd.DataFrame(x)

def c(df,names):
    return next((x for x in names if x in df.columns),None)

def ns(df,names,default=0.0):
    cc=c(df,names)
    return pd.to_numeric(df[cc],errors='coerce').fillna(default) if cc else pd.Series(default,index=df.index,dtype=float)

@st.cache_data(ttl=3600,show_spinner=False)
def load_stats(season):
    if nfl is None: raise RuntimeError('nflreadpy is not installed')
    for fn in [lambda:nfl.load_player_stats([season]),lambda:nfl.load_player_stats(seasons=[season]),lambda:nfl.load_player_stats(season)]:
        try:
            d=to_pd(fn())
            if len(d): return d
        except: pass
    raise RuntimeError('player stats unavailable')

@st.cache_data(ttl=1800,show_spinner=False)
def load_pbp(season):
    if nfl is None: raise RuntimeError('nflreadpy is not installed')
    for fn in [lambda:nfl.load_pbp([season]),lambda:nfl.load_pbp(seasons=[season]),lambda:nfl.load_pbp(season)]:
        try:
            d=to_pd(fn())
            if len(d): return d
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=900,show_spinner=False)
def load_schedule():
    if nfl is not None:
        for fn in [lambda:nfl.load_schedules(),lambda:nfl.load_schedules(seasons=True)]:
            try:
                d=to_pd(fn())
                if len(d): return d
            except: pass
    return pd.read_csv('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',low_memory=False)

def latest_stats(preferred):
    for y in [preferred,preferred-1,preferred-2]:
        try: return y,load_stats(y)
        except: pass
    raise RuntimeError('No player stats available')

def preseason_logs(season):
    pbp=load_pbp(season)
    if pbp.empty: return pd.DataFrame()
    typecol=c(pbp,['season_type','game_type'])
    if not typecol: return pd.DataFrame()
    pbp=pbp[pbp[typecol].astype(str).str.upper().isin(['PRE','PRESEASON'])].copy()
    if pbp.empty: return pd.DataFrame()
    gid=c(pbp,['game_id']); wk=c(pbp,['week'])
    if not gid: return pd.DataFrame()
    frames=[]
    specs=[
      ('rush',['rusher_player_name','rusher_name'],['rusher_player_id','rusher_id'],'rushing_yards',['rush_touchdown'],['rush_attempt']),
      ('rec',['receiver_player_name','receiver_name'],['receiver_player_id','receiver_id'],'receiving_yards',['pass_touchdown'],['pass_attempt']),
      ('pass',['passer_player_name','passer_name'],['passer_player_id','passer_id'],'passing_yards',[],['pass_attempt'])]
    for kind,names,ids,yardcol,tdnames,oppnames in specs:
        n=c(pbp,names); pid=c(pbp,ids); y=c(pbp,[yardcol]); td=c(pbp,tdnames); op=c(pbp,oppnames)
        if not n or not y: continue
        cols=[gid,n,y]+([pid] if pid else [])+([td] if td else [])+([op] if op else [])+([wk] if wk else [])
        t=pbp[cols].copy(); t=t[t[n].notna()]
        if t.empty: continue
        t['player_name']=t[n].astype(str); t['player_id']=t[pid].astype(str) if pid else t['player_name']
        keys=[gid,'player_id','player_name']+([wk] if wk else [])
        out=pd.DataFrame()
        if kind=='rush':
            t['rushing_yards']=pd.to_numeric(t[y],errors='coerce').fillna(0); t['rushing_tds']=pd.to_numeric(t[td],errors='coerce').fillna(0) if td else 0; t['carries']=pd.to_numeric(t[op],errors='coerce').fillna(0) if op else 1
            out=t.groupby(keys,as_index=False)[['rushing_yards','rushing_tds','carries']].sum(); out['receiving_yards']=0.;out['receiving_tds']=0.;out['targets']=0.;out['passing_yards']=0.;out['attempts']=0.
        elif kind=='rec':
            t['receiving_yards']=pd.to_numeric(t[y],errors='coerce').fillna(0); t['receiving_tds']=pd.to_numeric(t[td],errors='coerce').fillna(0) if td else 0; t['targets']=pd.to_numeric(t[op],errors='coerce').fillna(0) if op else 1
            out=t.groupby(keys,as_index=False)[['receiving_yards','receiving_tds','targets']].sum(); out['rushing_yards']=0.;out['rushing_tds']=0.;out['carries']=0.;out['passing_yards']=0.;out['attempts']=0.
        else:
            t['passing_yards']=pd.to_numeric(t[y],errors='coerce').fillna(0); t['attempts']=pd.to_numeric(t[op],errors='coerce').fillna(0) if op else 1
            out=t.groupby(keys,as_index=False)[['passing_yards','attempts']].sum(); out['rushing_yards']=0.;out['rushing_tds']=0.;out['carries']=0.;out['receiving_yards']=0.;out['receiving_tds']=0.;out['targets']=0.
        frames.append(out)
    if not frames: return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True)
    keys=[gid,'player_id','player_name']+([wk] if wk else [])
    nums=['rushing_yards','rushing_tds','carries','receiving_yards','receiving_tds','targets','passing_yards','attempts']
    return d.groupby(keys,as_index=False)[nums].sum()

def find_pre(d,name):
    if d.empty:return pd.DataFrame()
    x=d[d['player_name'].str.lower()==str(name).lower()]
    if len(x): return x
    last=str(name).split()[-1].lower()
    return d[d['player_name'].str.lower().str.endswith(last)]

def next_game(sched,team,season,include_pre):
    s=sched.copy()
    if 'season' in s: s=s[pd.to_numeric(s['season'],errors='coerce')==season]
    if 'game_type' in s:
        allowed=['REG','WC','DIV','CON','SB']+(['PRE'] if include_pre else [])
        s=s[s['game_type'].isin(allowed)]
    if 'gameday' not in s:return None
    s['_d']=pd.to_datetime(s['gameday'],errors='coerce').dt.date
    s=s[(s['_d']>=date.today())&((s['home_team']==team)|(s['away_team']==team))].sort_values('_d')
    if s.empty:return None
    g=s.iloc[0]; home=g['home_team']; away=g['away_team']
    return {'opp':away if home==team else home,'loc':'vs' if home==team else '@','date':g['gameday'],'type':g.get('game_type','')}

season=date.today().year if date.today().month>=7 else date.today().year-1
season=st.selectbox('NFL season',list(range(season,2022,-1)))
mode=st.selectbox('Data mode',['Blended (recommended)','Preseason only','Regular season only'])
pre_w=st.slider('Preseason weight %',10,50,25,5) if mode=='Blended (recommended)' else 0

with st.spinner('Loading NFL data...'):
    sy,stats=latest_stats(season); sched=load_schedule(); pre=preseason_logs(season)

n=c(stats,['player_display_name','player_name','display_name','name']); pid=c(stats,['player_id','gsis_id','nflverse_id']); teamc=c(stats,['recent_team','team','team_abbr']); posc=c(stats,['position','position_group','pos']); wk=c(stats,['week','week_num'])
pool=stats.copy()
if posc: pool=pool[pool[posc].astype(str).str.upper().isin(['QB','RB','WR','TE'])]
if pid:
    latest=pool.sort_values(wk) if wk else pool; latest=latest.groupby(pid,as_index=False).tail(1); latest['label']=latest[n].astype(str)
    if posc: latest['label']+=' — '+latest[posc].astype(str)
    if teamc: latest['label']+=' — '+latest[teamc].astype(str)
    sel=st.selectbox('Player',latest.sort_values('label')['label'].tolist()); r=latest[latest['label']==sel].iloc[0]; reg=pool[pool[pid]==r[pid]].copy(); pname=r[n]
else:
    pname=st.selectbox('Player',sorted(pool[n].dropna().astype(str).unique())); reg=pool[pool[n].astype(str)==pname].copy()
latestrow=reg.sort_values(wk).iloc[-1] if wk else reg.iloc[-1]; team=str(latestrow[teamc]) if teamc else ''; pos=str(latestrow[posc]) if posc else ''; ppre=find_pre(pre,pname)
match=next_game(sched,team,season,mode!='Regular season only')

m1,m2,m3=st.columns(3);m1.metric('Team',team or '—');m2.metric('Position',pos or '—');m3.metric('Preseason games',len(ppre))
if match: st.write(f"**Next matchup:** {team} {match['loc']} {match['opp']} — {'Preseason' if str(match['type']).upper()=='PRE' else 'Regular season'} — {match['date']}")
if ppre.empty and mode!='Regular season only': st.warning('No preseason play-by-play found for this player yet; using regular-season history.')

def reg_td(df,pos):
    t=ns(df,['rushing_tds','rush_tds'])+ns(df,['receiving_tds','rec_tds']); rush=ns(df,['carries','rushing_attempts']); targ=ns(df,['targets']) ; opp=targ if pos.upper() in ['WR','TE'] else rush+targ
    return float(t.mean()),float(t.tail(5).mean()),float(opp.mean())
def pre_td(df,pos):
    if df.empty:return 0.,0.
    t=df['rushing_tds']+df['receiving_tds'];opp=df['targets'] if pos.upper() in ['WR','TE'] else df['carries']+df['targets'];return float(t.mean()),float(opp.mean())
def reg_y(df,m):
    ym={'Rushing Yards':['rushing_yards'],'Receiving Yards':['receiving_yards'],'Passing Yards':['passing_yards']}[m];om={'Rushing Yards':['carries'],'Receiving Yards':['targets'],'Passing Yards':['attempts','passing_attempts']}[m];v=ns(df,ym);o=ns(df,om);return float(v.mean()),float(v.tail(5).mean()),float(v.std(ddof=1) if len(v)>1 else 10),float(o.mean())
def pre_y(df,m):
    if df.empty:return 0.,10.,0.
    y={'Rushing Yards':'rushing_yards','Receiving Yards':'receiving_yards','Passing Yards':'passing_yards'}[m];o={'Rushing Yards':'carries','Receiving Yards':'targets','Passing Yards':'attempts'}[m];v=pd.to_numeric(df[y],errors='coerce').fillna(0);op=pd.to_numeric(df[o],errors='coerce').fillna(0);return float(v.mean()),float(v.std(ddof=1) if len(v)>1 else max(v.mean()*.25,10)),float(op.mean())

tdtab,ytab=st.tabs(['🔥 Anytime TD','📏 Yardage Props'])
with tdtab:
    ra,rl5,ro=reg_td(reg,pos); pa,po=pre_td(ppre,pos)
    if mode=='Preseason only' and len(ppre): base=pa; normal=max(po,.5)
    elif mode=='Regular season only' or ppre.empty: base=.65*ra+.35*rl5; normal=max(ro,.5)
    else:
        w=pre_w/100;base=(1-w)*(.65*ra+.35*rl5)+w*pa;normal=(1-w)*max(ro,.5)+w*max(po,.5)
    a,b,c1=st.columns(3);a.metric('Regular TD/game',f'{ra:.2f}');b.metric('Preseason TD/game',f'{pa:.2f}');c1.metric('Model base',f'{base:.2f}')
    odds=st.number_input('Anytime TD odds',-1000,3000,150,5,key='tdo'); exp=st.number_input('Expected opportunities',.1,50.,float(round(normal,1)),.5,key='tdexp'); adj=st.slider('Role/red-zone adjustment %',-25,25,0,1,key='tda')
    lam=max(base*np.clip(exp/normal,.5,1.7)*(1+adj/100),0); prob=1-math.exp(-lam);book=american_to_prob(odds);edge=prob-book;fair=prob_to_american(prob);profit=float(odds) if odds>0 else 10000/abs(float(odds));ev=prob*profit-(1-prob)*100
    rng=np.random.default_rng(42); sim=float(np.mean(rng.poisson(lam,50000)>=1))
    st.divider();x1,x2=st.columns(2);x1.metric('TD probability',f'{prob:.1%}');x2.metric('Monte Carlo',f'{sim:.1%}');x3,x4=st.columns(2);x3.metric('Book implied',f'{book:.1%}');x4.metric('Model edge',f'{edge:+.1%}');x5,x6=st.columns(2);x5.metric('Fair odds',f'{fair:+d}');x6.metric('Expected TDs',f'{lam:.2f}');st.metric('EV / $100',f'${ev:+.2f}')
with ytab:
    m=st.selectbox('Market',['Rushing Yards','Receiving Yards','Passing Yards']); ra,rl5,rsd,ro=reg_y(reg,m); pa,psd,po=pre_y(ppre,m)
    if mode=='Preseason only' and len(ppre): base=pa;normal=max(po,.5);sd=psd
    elif mode=='Regular season only' or ppre.empty: base=.6*ra+.4*rl5;normal=max(ro,.5);sd=rsd
    else:
        w=pre_w/100;base=(1-w)*(.6*ra+.4*rl5)+w*pa;normal=(1-w)*max(ro,.5)+w*max(po,.5);sd=(1-w)*rsd+w*psd
    a,b,c1=st.columns(3);a.metric('Regular avg',f'{ra:.1f}');b.metric('Preseason avg',f'{pa:.1f}');c1.metric('Model base',f'{base:.1f}')
    line=st.number_input('Sportsbook line',.5,600.5,float(max(.5,round(base*2)/2 if base>0 else 50.5)),.5);side=st.radio('Side',['Over','Under'],horizontal=True);odds=st.number_input('American odds',-1000,3000,-110,5,key='yo');exp=st.number_input('Expected opportunities',.1,70.,float(round(normal,1)),.5,key='yexp');adj=st.slider('Matchup/role adjustment %',-20,20,0,1,key='ya')
    proj=max(base*np.clip(exp/normal,.55,1.6)*(1+adj/100),0);sd=max(sd,proj*.18,8);rng=np.random.default_rng(42);samples=np.maximum(rng.normal(proj,sd,50000),0);over=float(np.mean(samples>line));prob=over if side=='Over' else 1-over;book=american_to_prob(odds);edge=prob-book;fair=prob_to_american(prob);profit=float(odds) if odds>0 else 10000/abs(float(odds));ev=prob*profit-(1-prob)*100
    st.divider();y1,y2=st.columns(2);y1.metric('Projected yards',f'{proj:.1f}');y2.metric(f'{side} probability',f'{prob:.1%}');y3,y4=st.columns(2);y3.metric('Book implied',f'{book:.1%}');y4.metric('Model edge',f'{edge:+.1%}');y5,y6=st.columns(2);y5.metric('Fair odds',f'{fair:+d}');y6.metric('Simulation SD',f'{sd:.1f}');st.metric('EV / $100',f'${ev:+.2f}')

st.caption('Blended mode intentionally down-weights preseason results. Adjust Expected Opportunities to reflect expected preseason playing time.')
