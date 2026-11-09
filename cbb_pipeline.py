# Version 3: season-specific D1 membership filtering, retained schedule workload, and leak-safe model feature lists.
"""CBBD pregame feature builder. Python 3.10+."""
import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

BASE = 'https://api.collegebasketballdata.com'
METRICS = ['points_for','points_against','margin','possessions','pace','off_eff','def_eff',
           'efg','opp_efg','fg3_pct','ft_rate','oreb_pct','tov_pct','assists','rebounds',
           'steals','blocks','sos_win','sos_margin','sor','win']


def get_json(endpoint, params, cache_path, session, refresh=False):
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    for attempt in range(5):
        try:
            r = session.get(BASE + endpoint, params=params, timeout=75)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError(f'Expected list from {endpoint}: {str(data)[:150]}')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data))
            return data
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'API retries exhausted: {endpoint} {params}')


def fetch_season(season, session, cache_dir, refresh=False, chunk_days=14):
    # A season is labeled by its ending year. Broad date window includes postseason.
    begin = datetime(season - 1, 10, 1, tzinfo=timezone.utc)
    finish = datetime(season, 6, 1, tzinfo=timezone.utc)
    output = {}
    for endpoint, label in [('/games', 'games'), ('/games/teams', 'box'), ('/lines', 'lines')]:
        combined = []
        cursor = begin
        while cursor < finish:
            nxt = min(cursor + timedelta(days=chunk_days), finish)
            params = {'season': season, 'startDateRange': cursor.isoformat(),
                      'endDateRange': (nxt - timedelta(microseconds=1)).isoformat()}
            cache_path = Path(cache_dir) / str(season) / f'{label}_{cursor:%Y%m%d}.json'
            chunk = get_json(endpoint, params, cache_path, session, refresh)
            if len(chunk) >= 3000:
                raise RuntimeError(f'API cap reached for {label} {cursor}: reduce --chunk-days')
            combined.extend(chunk)
            cursor = nxt
        output[label] = combined
        print(f'{season} {label}: {len(combined)} records')
    return output


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def nested(d, *keys):
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def ratio(a, b, scale=1.0):
    a, b = num(a), num(b)
    return scale * a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def game_box_features(row):
    s, opp = row.get('teamStats') or {}, row.get('opponentStats') or {}
    pts = num(nested(s, 'points', 'total'))
    opp_pts = num(nested(opp, 'points', 'total'))
    poss = num(s.get('possessions'))
    # Estimate possessions only when source count unavailable.
    if not np.isfinite(poss):
        fga = num(nested(s, 'fieldGoals', 'attempted'))
        fta = num(nested(s, 'freeThrows', 'attempted'))
        oreb = num(nested(s, 'rebounds', 'offensive'))
        tov = num(nested(s, 'turnovers', 'total'))
        if all(np.isfinite(x) for x in (fga, fta, oreb, tov)):
            poss = fga - oreb + tov + 0.475 * fta
    fgm = num(nested(s, 'fieldGoals', 'made'))
    fga = num(nested(s, 'fieldGoals', 'attempted'))
    fg3m = num(nested(s, 'threePointFieldGoals', 'made'))
    opp_fgm = num(nested(opp, 'fieldGoals', 'made'))
    opp_fga = num(nested(opp, 'fieldGoals', 'attempted'))
    opp_fg3m = num(nested(opp, 'threePointFieldGoals', 'made'))
    return {
        'points_for': pts, 'points_against': opp_pts, 'margin': pts - opp_pts,
        'win': float(pts > opp_pts) if np.isfinite(pts) and np.isfinite(opp_pts) else np.nan,
        'possessions': poss, 'pace': num(row.get('pace')),
        'off_eff': ratio(pts, poss, 100), 'def_eff': ratio(opp_pts, poss, 100),
        'efg': ratio(fgm + .5 * fg3m, fga),
        'opp_efg': ratio(opp_fgm + .5 * opp_fg3m, opp_fga),
        'fg3_pct': ratio(fg3m, nested(s, 'threePointFieldGoals', 'attempted')),
        'ft_rate': ratio(nested(s, 'freeThrows', 'attempted'), fga),
        'oreb_pct': ratio(nested(s, 'rebounds', 'offensive'),
                          num(nested(s, 'rebounds', 'offensive')) + num(nested(opp, 'rebounds', 'defensive'))),
        'tov_pct': ratio(nested(s, 'turnovers', 'total'), poss),
        'assists': num(s.get('assists')), 'rebounds': num(nested(s, 'rebounds', 'total')),
        'steals': num(s.get('steals')), 'blocks': num(s.get('blocks')),
    }


def index_box_scores(box_rows):
    return {(int(r['gameId']), int(r['teamId'])): game_box_features(r)
            for r in box_rows if r.get('gameId') is not None and r.get('teamId') is not None}


def index_lines(line_rows):
    return {int(r['gameId']): r.get('lines') or [] for r in line_rows if r.get('gameId') is not None}


def line_columns(lines):
    # CBBD spread is home-team handicap; predicted home margin = -spread.
    valid = [x for x in lines if isinstance(x, dict)]
    valid.sort(key=lambda x: (str(x.get('provider') or '')).lower())
    if not valid:
        return {'market_provider': None, 'market_home_spread': np.nan,
                'market_home_margin': np.nan, 'market_open_spread': np.nan,
                'market_total': np.nan, 'market_provider_count': 0}
    x = valid[0]
    spread = num(x.get('spread'))
    return {'market_provider': x.get('provider'), 'market_home_spread': spread,
            'market_home_margin': -spread if np.isfinite(spread) else np.nan,
            'market_open_spread': num(x.get('spreadOpen')),
            'market_total': num(x.get('overUnder')),
            'market_provider_count': len(valid)}


def summarize(team, history, date, schedule_history=None):
    h = history.get(team, [])
    result = {'games_played': len(h)}
    for metric in METRICS:
        values = [x.get(metric, np.nan) for x in h]
        good = [v for v in values if np.isfinite(v)]
        result[f'avg_{metric}'] = float(np.mean(good)) if good else np.nan
        last = [v for v in values[-5:] if np.isfinite(v)]
        result[f'last5_{metric}'] = float(np.mean(last)) if last else np.nan
    played = (schedule_history or {}).get(team, [])
    result['rest_days'] = (date - played[-1]).days if played else np.nan
    result['games_last_7_days'] = sum(0 < (date - d).days <= 7 for d in played)
    result['all_games_played'] = len(played)
    # SOS: average pregame strength of opponents encountered, computed at the
    # time each opponent was faced (strictly lagged). SOR is win quality:
    # average (win * (0.5 + opponent pregame win percentage)).
    for metric in ('sos_win','sos_margin','sor'):
        vals = [x.get(metric, np.nan) for x in h]
        vals = [v for v in vals if np.isfinite(v)]
        result[f'avg_{metric}'] = float(np.mean(vals)) if vals else np.nan
    return result


def season_d1_membership(games):
    """Infer season-specific D1 membership from nonempty conference affiliation.

    A team is D1 if it is shown with a conference in ANY game this season.
    This avoids treating an occasional missing conference as proof of non-D1.
    Independent D1 teams without any conference affiliation need manual review.
    Membership is static metadata, never performance information.
    """
    members = set()
    for g in games:
        for side in ('home', 'away'):
            tid = g.get(side + 'TeamId')
            conf = g.get(side + 'Conference')
            if tid is not None and conf is not None and str(conf).strip():
                members.add(int(tid))
    return members


def build_pregame_rows(raw, season):
    games = [g for g in raw['games'] if g.get('status') == 'final'
             and g.get('homePoints') is not None and g.get('awayPoints') is not None]
    games.sort(key=lambda g: (str(g['startDate'])[:10], str(g['startDate']), g['id']))
    boxes, lines = index_box_scores(raw['box']), index_lines(raw['lines'])
    d1_ids = season_d1_membership(raw['games'])
    print(f'{season}: inferred {len(d1_ids)} D1 team IDs from conference membership')
    history = defaultdict(list)
    schedule_history = defaultdict(list)
    rows, audit, exclusions = [], [], []
    # Batch by calendar date to avoid using unknown intraday game ordering.
    for date_key, group in __import__('itertools').groupby(games, key=lambda g: str(g['startDate'])[:10]):
        group = list(group)
        date = datetime.strptime(date_key, '%Y-%m-%d')
        pending = []
        schedule_pending = []
        for g in group:
            gid, home, away = int(g['id']), int(g['homeTeamId']), int(g['awayTeamId'])
            hs = summarize(home, history, date, schedule_history)
            as_ = summarize(away, history, date, schedule_history)
            home_box, away_box = boxes.get((gid, home)), boxes.get((gid, away))
            # Every completed game affects rest and workload, even when its
            # box score is unavailable or the opponent is outside D1.
            schedule_pending.extend([(home, date), (away, date)])
            # A game is a model row ONLY when both teams are season D1 members.
            # Non-D1 dates still count toward rest and recent workload.
            if home not in d1_ids or away not in d1_ids:
                exclusions.append({'game_id': gid, 'season': season,
                                   'date': g['startDate'],
                                   'home_team': g.get('homeTeam'),
                                   'away_team': g.get('awayTeam'),
                                   'reason': 'non_d1_or_unverified_membership',
                                   'home_d1': int(home in d1_ids),
                                   'away_d1': int(away in d1_ids)})
                continue
            # Also require a usable box score for both D1 teams.
            if not (home_box and away_box and
                    np.isfinite(home_box['points_for']) and
                    np.isfinite(away_box['points_for']) and
                    np.isfinite(home_box['possessions']) and
                    np.isfinite(away_box['possessions'])):
                exclusions.append({'game_id': gid, 'season': season,
                                   'date': g['startDate'],
                                   'home_team': g.get('homeTeam'),
                                   'away_team': g.get('awayTeam'),
                                   'reason': 'missing_required_team_box_stats',
                                   'home_d1': 1, 'away_d1': 1})
                continue
            record = {'game_id': gid, 'season': season, 'date': g['startDate'],
                      'home_team_id': home, 'away_team_id': away,
                      'home_team': g.get('homeTeam'), 'away_team': g.get('awayTeam'),
                      'home_conference': g.get('homeConference'), 'away_conference': g.get('awayConference'),
                      'neutral_site': int(bool(g.get('neutralSite'))),
                      'conference_game': int(bool(g.get('conferenceGame'))),
                      'home_points': num(g['homePoints']), 'away_points': num(g['awayPoints']),
                      'target_home_margin': num(g['homePoints']) - num(g['awayPoints']),
                      'has_home_box': int(home_box is not None), 'has_away_box': int(away_box is not None)}
            for k in hs:
                record['home_' + k], record['away_' + k] = hs[k], as_[k]
                if k not in ('games_played', 'all_games_played'):
                    a, b = hs[k], as_[k]
                    record['diff_' + k] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
            record.update(line_columns(lines.get(gid, [])))
            rows.append(record)
            audit.extend({'game_id': gid, 'provider': x.get('provider'),
                          'spread': x.get('spread'), 'spread_open': x.get('spreadOpen'),
                          'total': x.get('overUnder'), 'total_open': x.get('overUnderOpen'),
                          'home_moneyline': x.get('homeMoneyline'), 'away_moneyline': x.get('awayMoneyline')}
                         for x in lines.get(gid, []) if isinstance(x, dict))
            # Update team histories after all games from this date have been predicted.
            opp_win_away = as_.get('avg_win', np.nan)
            opp_win_home = hs.get('avg_win', np.nan)
            opp_margin_away = as_.get('avg_margin', np.nan)
            opp_margin_home = hs.get('avg_margin', np.nan)
            hb = dict(home_box, date=date, sos_win=opp_win_away,
                      sos_margin=opp_margin_away,
                      sor=home_box['win'] * (.5 + opp_win_away) if np.isfinite(opp_win_away) else np.nan)
            ab = dict(away_box, date=date, sos_win=opp_win_home,
                      sos_margin=opp_margin_home,
                      sor=away_box['win'] * (.5 + opp_win_home) if np.isfinite(opp_win_home) else np.nan)
            pending.extend([(home, hb), (away, ab)])
        for team, item in pending:
            history[team].append(item)
        for team, played_date in schedule_pending:
            schedule_history[team].append(played_date)
    excluded_cols = ['game_id','season','date','home_team','away_team','reason','home_d1','away_d1']
    return pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(exclusions, columns=excluded_cols)


def normalize_seasons(frames, output_dir, train_end=2025):
    """Fit imputer/scaler only on seasons <= train_end, transform all seasons."""
    exclude = {'game_id','season','date','home_team_id','away_team_id','home_team',
               'away_team','home_conference','away_conference','home_points','away_points',
               'target_home_margin','market_provider','market_home_spread','market_home_margin',
               'market_open_spread','market_total','market_provider_count'}
    # These are constants after eligibility filtering and carry no predictive value.
    exclude.update({'has_home_box', 'has_away_box'})
    features = [c for c in frames[min(frames)].columns if c not in exclude
                and pd.api.types.is_numeric_dtype(frames[min(frames)][c])]
    training_years = [year for year in frames if year <= train_end]
    if not training_years:
        raise ValueError('No training seasons: --train-end must be >= --start')
    train = pd.concat([frames[year][features] for year in training_years], ignore_index=True)
    # All-missing columns are retained by SimpleImputer in recent sklearn versions;
    # explicitly remove them for compatibility with older versions.
    active = [c for c in features if train[c].notna().any()]
    missing = [c for c in active if train[c].isna().any()]
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train[active])
    scaler.fit(x_train)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'features': active,
                 'missing_indicator_features': missing, 'fit_through_season': train_end},
                output_dir / 'preprocessing.joblib')
    for year, df in frames.items():
        out = df.copy()
        vals = scaler.transform(imputer.transform(df[active]))
        for i, c in enumerate(active):
            out['z_' + c] = vals[:, i]
        for c in missing:
            out['missing_' + c] = df[c].isna().astype(int)
        out.to_csv(output_dir / f'games_{year-1}_{year}.csv', index=False)
        print(f'Wrote games_{year-1}_{year}.csv ({len(out)} games)')
    (output_dir / 'feature_columns.json').write_text(json.dumps({
        'normalized_features': ['z_' + c for c in active],
        'missing_indicators': ['missing_' + c for c in missing],
        'target': 'target_home_margin',
        'training_seasons': sorted(training_years),
        'evaluation_seasons': sorted(year for year in frames if year > train_end),
        'market_columns': sorted(c for c in exclude if c.startswith('market_')),
        'note': 'Do not use raw scores, game outcome, market columns, or IDs as predictors.'}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=2017, help='Ending year of first season')
    p.add_argument('--end', type=int, default=2026, help='Ending year of last season')
    p.add_argument('--train-end', type=int, default=2025)
    p.add_argument('--output', default='cbb_data')
    p.add_argument('--chunk-days', type=int, default=14)
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--audit-only', action='store_true', help='Reserved; not implemented')
    args = p.parse_args()
    if args.audit_only:
        p.error('--audit-only is not implemented')
    if args.start > args.end:
        p.error('--start must not exceed --end')
    token = os.environ.get('CBBD_API_KEY')
    if not token:
        p.error('Set CBBD_API_KEY environment variable first')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {token}', 'Accept': 'application/json'})
    frames = {}
    for year in range(args.start, args.end + 1):
        raw = fetch_season(year, session, out / 'cache', args.refresh, args.chunk_days)
        frames[year], lines, exclusions = build_pregame_rows(raw, year)
        if frames[year].empty:
            raise RuntimeError(f'No completed games in {year}')
        lines.to_csv(out / f'lines_all_providers_{year-1}_{year}.csv', index=False)
        exclusions.to_csv(out / f'excluded_games_{year-1}_{year}.csv', index=False)
        print(f'{year}: retained {len(frames[year])}, excluded {len(exclusions)} non-D1/unverified/incomplete games')
        frames[year].to_csv(out / f'raw_pregame_{year-1}_{year}.csv', index=False)
    normalize_seasons(frames, out, args.train_end)


if __name__ == '__main__':
    main()
