import streamlit as st
import pandas as pd
from collections import deque, defaultdict

st.set_page_config(page_title="Sticker Trade Automator", layout="wide")
st.title("🎟️ Sticker Trade Automator (In-browser, no Excel required)")

# -----------------------
# Helpers: Hopcroft-Karp
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
# UI: sticker IDs inputs
# -----------------------
st.markdown("**Step 1 — Enter this week's sticker #s 1st, THEN click CLEAR DATA before entering info**")
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
# Create initial DataFrame stored in session_state
def make_empty_df():
    stickers = st.session_state["stickers"]
    cols = ["Name"] + [f"DUP_{s}" for s in stickers] + [f"NEED_{s}" for s in stickers]
    df = pd.DataFrame(columns=cols)
    # create 50 rows
    for _ in range(50):
        df = pd.concat([df, pd.DataFrame([{}])], ignore_index=True)
    df.fillna("", inplace=True)
    return df

if "grid_df" not in st.session_state:
    st.session_state["grid_df"] = make_empty_df()

# show experimental data editor (editable grid)
edited = st.data_editor(st.session_state["grid_df"], num_rows="dynamic", use_container_width=True)
st.session_state["grid_df"] = edited

# Buttons
col_run, col_clear = st.columns([1,1])
with col_run:
    run_pressed = st.button("Run Trades", type="primary")
with col_clear:
    clear_pressed = st.button("Clear Data")

# Clear logic
if clear_pressed:
    st.session_state["grid_df"] = make_empty_df()
    st.rerun()

# -----------------------
# Core: convert grid to internal structures and run matching
# -----------------------
def run_matching_from_grid(df, sticker_ids):
    # normalize names and limit to first 50 rows
    df2 = df.copy().reset_index(drop=True).head(50)
    names = df2["Name"].fillna("").astype(str).tolist()
    n = len(names)

    # Build gives and needs maps using sticker_ids ordering
    gives = defaultdict(set)
    needs = defaultdict(lambda: defaultdict(int))  # counts for multiplicity

    for i in range(n):
        if names[i].strip() == "":
            continue
        for j, sid in enumerate(sticker_ids):
            dup_col = f"DUP_{sid}"
            need_col = f"NEED_{sid}"
            # some users may type '1' as text; accept '1' or 1
            val_dup = df2.at[i, dup_col] if dup_col in df2.columns else ""
            val_need = df2.at[i, need_col] if need_col in df2.columns else ""
            if str(val_dup).strip() == "1":
                gives[i+1].add(str(sid))
            if str(val_need).strip() == "1":
                needs[i+1][str(sid)] += 1

    # Build giver slots (each duplicate item is a slot)
    giver_slots = []  # tuples (personIndex, sticker)
    for p in range(1, n+1):
        for sid in gives[p]:
            giver_slots.append((p, sid))
    gcount = len(giver_slots)

    # Build need_counts dict (mutable) for remaining needs
    need_counts = {}
    for p in range(1, n+1):
        for sid, cnt in needs[p].items():
            need_counts[(p, sid)] = cnt

    trades = []  # (giverPerson, receiverPerson, sticker)
    giver_used = [False] * (n + 1)
    received_once = [False] * (n + 1)

    # Phase 1: ensure everyone with any need receives once
    progress = True
    while progress:
        progress = False
        # left map = indices of giver_slots where giver not used
        Lmap = [i+1 for i, (gp, gs) in enumerate(giver_slots) if not giver_used[gp]]
        if not Lmap:
            break

        # receivers allowed = persons who had at least one original need and haven't received once
        receivers_allowed = set()
        for p in range(1, n+1):
            # originally had needs?
            if sum(needs[p].values()) > 0 and not received_once[p]:
                receivers_allowed.add(p)
        if not receivers_allowed:
            break

        # build adjacency mapping 1..len(Lmap) -> dense right indices 1..Rsize
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

        Lsize = len(Lmap)
        Rsize = len(right_list)
        matchL, matchR = hopcroft_karp(adj, Lsize, Rsize)

        # apply matches
        for u in range(1, Lsize+1):
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

    # Fairness nudge: try to match givers who gave but haven't received yet
    givers_who_gave = set([g for g,_,_ in trades])
    givers_received = set([r for _,r,_ in trades])
    fairness_targets = [p for p in givers_who_gave if p not in givers_received and any(need_counts.get((p,s),0)>0 for s in needs[p])]
    if fairness_targets:
        Lmap2 = [i+1 for i, (gp,gs) in enumerate(giver_slots) if not giver_used[gp]]
        if Lmap2:
            right_list2 = sorted(fairness_targets)
            rindex2 = {person: idx+1 for idx, person in enumerate(right_list2)}
            adj2 = {}
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

    # Phase 2: extras allowed — build remaining giver slots and remaining need units
    Lmap3 = [i+1 for i,(gp,gs) in enumerate(giver_slots) if not giver_used[gp]]
    if Lmap3:
        need_slots = []
        for p in range(1, n+1):
            for (sid_count_key), cnt in list(need_counts.items()):
                pass
        # build structured need_slots from need_counts
        need_slots = []
        for p in range(1, n+1):
            for sid in [s for s in needs[p].keys()]:
                cnt = need_counts.get((p, sid), 0)
                for _ in range(cnt):
                    need_slots.append((p, sid))
        R3 = len(need_slots)
        if R3 > 0:
            adj3 = {u: [] for u in range(1, len(Lmap3)+1)}
            for ui, gi in enumerate(Lmap3, start=1):
                gp, gs = giver_slots[gi-1]
                for ri, (rp, rs) in enumerate(need_slots, start=1):
                    if rp != gp and rs == gs:
                        adj3[ui].append(ri)
            matchL3, matchR3 = hopcroft_karp(adj3, len(Lmap3), R3)
            for u in range(1, len(Lmap3)+1):
                v = matchL3[u]
                if v != 0:
                    gi = Lmap3[u-1]
                    gp, gs = giver_slots[gi-1]
                    rp, rs = need_slots[v-1]
                    if not giver_used[gp]:
                        trades.append((gp, rp, gs))
                        giver_used[gp] = True

    # Convert trades to readable form
    trades_out = []
    for gp, rp, sid in trades:
        gname = names[gp-1] if gp-1 < len(names) else f"Person{gp}"
        rname = names[rp-1] if rp-1 < len(names) else f"Person{rp}"
        trades_out.append((gname, sid, rname))

    # Unmatched lists
    unmatched_givers = []
    unmatched_receivers = []
    for p in range(1, n+1):
        if gives.get(p) and not any(t[0]==names[p-1] for t in trades_out):
            unmatched_givers.append(names[p-1])
        had_need = sum(needs[p].values()) > 0
        if had_need and not any(t[2]==names[p-1] for t in trades_out):
            unmatched_receivers.append(names[p-1])

    return trades_out, unmatched_givers, unmatched_receivers

# -----------------------
# Run when button pressed
# -----------------------
if run_pressed:
    sticker_ids = sticker_inputs
    df_grid = st.session_state["grid_df"]
    trades_out, unmatched_givers, unmatched_receivers = run_matching_from_grid(df_grid, sticker_ids)

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
