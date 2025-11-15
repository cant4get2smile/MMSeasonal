import streamlit as st
import pandas as pd
from collections import deque, defaultdict

st.set_page_config(page_title="Sticker Trade Automator", layout="wide")
st.title("🎟️ Sticker Trade Automator (In-browser, no Excel required)")

# -----------------------
# Helpers: Hopcroft-Karp (for max matching)
# -----------------------
def hopcroft_karp(adj, Lsize, Rsize):
    INF = 10**9
    matchL = [0] * (Lsize + 1)
    matchR = [0] * (Rsize + 1)
    dist = [0] * (Lsize + 1)

    def bfs():
        q = deque()
        for u in range(1, Lsize+1):
            if matchL[u] == 0:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if matchR[v] == 0:
                    found = True
                else:
                    if dist[matchR[v]] == INF:
                        dist[matchR[v]] = dist[u] + 1
                        q.append(matchR[v])
        return found

    def dfs(u):
        for v in adj.get(u, []):
            if matchR[v] == 0 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v])):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(1, Lsize+1):
            if matchL[u] == 0:
                dfs(u)
    return matchL, matchR

# -----------------------
# UI step 1: sticker IDs
# -----------------------
st.markdown("**Step 1 — Enter this week's sticker IDs (these label the DUPS and NEEDS columns)**")
cols = st.columns(5)
default_stickers = st.session_state.get("stickers", ["3","6","14","16","21"])
sticker_inputs = []
for i, col in enumerate(cols):
    sticker_inputs.append(col.text_input(f"Sticker {i+1}", value=default_stickers[i], key=f"st{i}"))
st.session_state["stickers"] = sticker_inputs

# -----------------------
# Build / show editable grid
# -----------------------
st.markdown("**Step 2 — Enter names and mark DUPS / NEEDS with `1` (up to 50 rows)**")

def make_empty_df():
    stickers = st.session_state["stickers"]
    cols = ["Name"] + [f"DUP_{s}" for s in stickers] + [f"NEED_{s}" for s in stickers]
    df = pd.DataFrame(columns=cols)
    # create 50 blank rows
    df = pd.concat([df, pd.DataFrame([{} for _ in range(50)])], ignore_index=True)
    df.fillna("", inplace=True)
    return df

# ensure session grid exists
if "grid_df" not in st.session_state:
    st.session_state["grid_df"] = make_empty_df()

# show the editor, but *do not* immediately overwrite session state
# We'll parse the returned "edited" value when Run Trades is pressed.
edited_grid = st.data_editor(
    st.session_state["grid_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="grid_editor"
)

# Buttons: Run Trades and Clear Data
col_run, col_clear = st.columns([1,1])
run_pressed = col_run.button("Run Trades")
clear_pressed = col_clear.button("Clear Data")

if clear_pressed:
    st.session_state["grid_df"] = make_empty_df()
    st.experimental_rerun()  # or st.rerun() depending on Streamlit version

# -----------------------
# Parser: convert data_editor result into a clean DataFrame
# -----------------------
def parse_edited(edited):
    """
    edited can be:
      - a pandas DataFrame (common)
      - a dict with keys 'edited_rows' and 'added_rows' (Streamlit variations)
      - a list of dicts
    Return a DataFrame with consistent columns.
    """
    try:
        if isinstance(edited, pd.DataFrame):
            df = edited.copy()
        elif isinstance(edited, dict):
            # combine edited_rows + added_rows if present
            rows = []
            edited_rows = edited.get("edited_rows", {})
            added_rows = edited.get("added_rows", [])
            # edited_rows might be dict of rowindex -> rowdict
            if isinstance(edited_rows, dict):
                rows.extend(list(edited_rows.values()))
            elif isinstance(edited_rows, list):
                rows.extend(edited_rows)
            # added_rows typically is a list
            if isinstance(added_rows, list):
                rows.extend(added_rows)
            # If nothing, try to build from dict-of-lists
            if not rows:
                try:
                    df = pd.DataFrame.from_dict(edited)
                    df.fillna("", inplace=True)
                    return df
                except Exception:
                    return st.session_state["grid_df"]
            df = pd.DataFrame(rows)
        elif isinstance(edited, list):
            df = pd.DataFrame(edited)
        else:
            return st.session_state["grid_df"]
        df.fillna("", inplace=True)
        # ensure all expected columns exist (Name + DUP_ + NEED_)
        expected_cols = ["Name"] + [f"DUP_{s}" for s in st.session_state["stickers"]] + [f"NEED_{s}" for s in st.session_state["stickers"]]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""  # fill missing columns
        # keep column order
        df = df[expected_cols]
        return df
    except Exception as e:
        st.error(f"Error parsing grid: {e}")
        return st.session_state["grid_df"]

# -----------------------
# Matching logic (same logic as your VBA app)
# -----------------------
def run_matching_from_grid(df, sticker_ids):
    # df is expected to have columns: Name, DUP_<id> x5, NEED_<id> x5
    df2 = df.copy().reset_index(drop=True).head(50)
    names = df2["Name"].fillna("").astype(str).tolist()
    n = len(names)
    # Build gives and needs maps
    gives = defaultdict(set)
    needs = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        if names[i].strip() == "":
            continue
        for j, sid in enumerate(sticker_ids):
            dup_col = f"DUP_{sid}"
            need_col = f"NEED_{sid}"
            val_dup = df2.at[i, dup_col] if dup_col in df2.columns else ""
            val_need = df2.at[i, need_col] if need_col in df2.columns else ""
            if str(val_dup).strip() == "1":
                gives[i+1].add(str(sid))
            if str(val_need).strip() == "1":
                needs[i+1][str(sid)] += 1

    # Build giver slots
    giver_slots = []
    for p in range(1, n+1):
        for sid in gives[p]:
            giver_slots.append((p, sid))

    need_counts = {}
    for p in range(1, n+1):
        for sid, cnt in needs[p].items():
            need_counts[(p, sid)] = cnt

    trades = []
    giver_used = [False] * (n + 1)
    received_once = [False] * (n + 1)

    # Phase 1: ensure each person with needs receives once (if possible)
    progress = True
    while progress:
        progress = False
        Lmap = [i+1 for i, (gp, gs) in enumerate(giver_slots) if not giver_used[gp]]
        if not Lmap:
            break
        receivers_allowed = set()
        for p in range(1, n+1):
            if sum(needs[p].values()) > 0 and not received_once[p]:
                receivers_allowed.add(p)
        if not receivers_allowed:
            break
        right_list = sorted(list(receivers_allowed))
        rindex = {person: idx+1 for idx, person in enumerate(right_list)}
        adj = {}
        for iL, gi in enumerate(Lmap, start=1):
            gp, gs = giver_slots[gi-1]
            adj[iL] = []
            for rp in right_list:
                if rp != gp and need_counts.get((rp, gs), 0) > 0:
                    adj[iL].append(rindex[rp])
        if not any(adj.values()):
            break
        matchL, matchR = hopcroft_karp(adj, len(Lmap), len(right_list))
        for u in range(1, len(Lmap)+1):
            v = matchL[u]
            if v != 0:
                gi = Lmap[u-1]
                gp, gs = giver_slots[gi-1]
                rp = right_list[v-1]
                if (not giver_used[gp]) and (not received_once[rp]) and need_counts.get((rp, gs),0) > 0:
                    trades.append((gp, rp, gs))
                    giver_used[gp] = True
                    received_once[rp] = True
                    need_counts[(rp, gs)] -= 1
                    progress = True

    # Fairness nudge
    givers_who_gave = set([g for g,_,_ in trades])
    givers_received = set([r for _,r,_ in trades])
    fairness_targets = [p for p in givers_who_gave if p not in givers_received and any(need_counts.get((p,s),0)>0 for s in needs[p])]
    if fairness_targets:
        Lmap2 = [i+1 for i,(gp,gs) in enumerate(giver_slots) if not giver_used[gp]]
        if Lmap2:
            right_list2 = sorted(fairness_targets)
            adj2 = {}
            rindex2 = {person: idx+1 for idx, person in enumerate(right_list2)}
            for iL, gi in enumerate(Lmap2, start=1):
                gp, gs = giver_slots[gi-1]
                adj2[iL] = []
                for rp in right_list2:
                    if rp != gp and need_counts.get((rp, gs),0) > 0:
                        adj2[iL].append(rindex2[rp])
            if any(adj2.values()):
                matchL2, matchR2 = hopcroft_karp(adj2, len(Lmap2), len(right_list2))
                for u in range(1, len(Lmap2)+1):
                    v = matchL2[u]
                    if v != 0:
                        gi = Lmap2[u-1]
                        gp, gs = giver_slots[gi-1]
                        rp = right_list2[v-1]
                        if (not giver_used[gp]) and need_counts.get((rp, gs),0) > 0:
                            trades.append((gp, rp, gs))
                            giver_used[gp] = True
                            received_once[rp] = True
                            need_counts[(rp, gs)] -= 1

    # Phase 2: allow extras to maximize trades
    Lmap3 = [i+1 for i,(gp,gs) in enumerate(giver_slots) if not giver_used[gp]]
    if Lmap3:
        need_slots = []
        for p in range(1, n+1):
            for sid in [s for s in needs[p].keys()]:
                cnt = need_counts.get((p, sid), 0)
                for _ in range(cnt):
                    need_slots.append((p, sid))
        if need_slots:
            adj3 = {u: [] for u in range(1, len(Lmap3)+1)}
            for ui, gi in enumerate(Lmap3, start=1):
                gp, gs = giver_slots[gi-1]
                for ri, (rp, rs) in enumerate(need_slots, start=1):
                    if rp != gp and rs == gs:
                        adj3[ui].append(ri)
            matchL3, matchR3 = hopcroft_karp(adj3, len(Lmap3), len(need_slots))
            for u in range(1, len(Lmap3)+1):
                v = matchL3[u]
                if v != 0:
                    gi = Lmap3[u-1]
                    gp, gs = giver_slots[gi-1]
                    rp, rs = need_slots[v-1]
                    if not giver_used[gp]:
                        trades.append((gp, rp, gs))
                        giver_used[gp] = True

    # Format output
    trades_out = []
    for gp, rp, sid in trades:
        gname = df2.at[gp-1, "Name"] if gp-1 < len(df2) else f"Person{gp}"
        rname = df2.at[rp-1, "Name"] if rp-1 < len(df2) else f"Person{rp}"
        trades_out.append((gname, sid, rname))

    # Unmatched lists
    unmatched_givers = []
    unmatched_receivers = []
    for p in range(1, n+1):
        if gives.get(p) and not any(t[0]==df2.at[p-1, "Name"] for t in trades_out):
            unmatched_givers.append(df2.at[p-1, "Name"])
        had_need = sum(needs[p].values()) > 0
        if had_need and not any(t[2]==df2.at[p-1, "Name"] for t in trades_out):
            unmatched_receivers.append(df2.at[p-1, "Name"])

    return trades_out, unmatched_givers, unmatched_receivers

# -----------------------
# When Run Trades is pressed: parse edited grid and run matching
# -----------------------
if run_pressed:
    sticker_ids = st.session_state["stickers"]
    parsed_df = parse_edited(edited_grid)
    # Save parsed grid into session so user can clear or view it later if needed
    st.session_state["grid_df"] = parsed_df
    trades_out, unmatched_givers, unmatched_receivers = run_matching_from_grid(parsed_df, sticker_ids)

    # Show results
    st.subheader("✅ Trade Results")
    if trades_out:
        trade_df = pd.DataFrame(trades_out, columns=["Giver","Sticker","Receiver"])
        trade_df["Summary"] = trade_df["Giver"] + " gives " + trade_df["Sticker"].astype(str) + " to " + trade_df["Receiver"]
        st.dataframe(trade_df[["Summary"]], use_container_width=True)
    else:
        st.info("No trades found.")

    st.subheader("⚠️ Unmatched Participants")
    c1, c2 = st.columns(2)
    c1.markdown("**Unmatched Givers**")
    c1.write("\n".join(unmatched_givers) if unmatched_givers else "None")
    c2.markdown("**Unmatched Receivers**")
    c2.write("\n".join(unmatched_receivers) if unmatched_receivers else "None")

    st.success(f"Trades: {len(trades_out)}  |  Unmatched Givers: {len(unmatched_givers)}  |  Unmatched Receivers: {len(unmatched_receivers)}")
