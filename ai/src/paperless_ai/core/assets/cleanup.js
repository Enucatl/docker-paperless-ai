const status = document.querySelector("#status");
const pairs = document.querySelector("#pairs");
const output = document.querySelector("#plan");

async function refresh() {
  const response = await fetch("api/plan");
  if (!response.ok) {
    status.textContent = await response.text();
    return;
  }
  const current = await response.json();
  const recorded = new Map(
    current.review_decisions.map((x) => [x.pair_key, x.decision]),
  );
  status.textContent = `${current.cleanup_mode} cleanup · snapshot max ID ${current.scanned_max_correspondent_id} · ${current.review_candidates.length} actionable review pair(s) · ${recorded.size} decision(s) recorded`;
  pairs.replaceChildren();
  for (const pair of current.review_candidates) {
    const card = document.createElement("section");
    card.className = "card";
    const selected = recorded.get(pair.pair_key);
    const left = document.createElement("strong");
    left.textContent = pair.left_name;
    const right = document.createElement("strong");
    right.textContent = pair.right_name;
    const reason = document.createElement("p");
    reason.className = "muted";
    reason.textContent = `${pair.reason || ""} · candidate score ${pair.candidate_score ?? "—"}`;
    card.append(
      left,
      ` (${pair.left_members.join(", ")}) ↔ `,
      right,
      ` (${pair.right_members.join(", ")})`,
      reason,
    );
    for (const choice of ["approve", "reject"]) {
      const button = document.createElement("button");
      button.textContent =
        selected === choice
          ? choice === "approve"
            ? "Approved"
            : "Rejected"
          : choice;
      button.className =
        selected === choice
          ? choice === "approve"
            ? "approved"
            : "rejected"
          : "";
      button.onclick = async () => {
        const response = await fetch("api/decisions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ pair_key: pair.pair_key, decision: choice }),
        });
        if (!response.ok) {
          alert((await response.json()).detail || "Unable to record decision");
          return;
        }
        status.textContent = `Decision recorded: ${choice}`;
        await showPlan();
        await refresh();
      };
      card.append(button);
    }
    pairs.append(card);
  }
  await showPlan();
}

async function showPlan() {
  const response = await fetch("api/reviewed-plan");
  output.textContent = response.ok
    ? JSON.stringify(await response.json(), null, 2)
    : await response.text();
}

document.querySelector("#apply").onclick = async () => {
  const response = await fetch("api/apply", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      confirmation: document.querySelector("#confirmation").value,
    }),
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { detail: text };
  }
  if (!response.ok) {
    status.textContent = body.detail || "Apply failed";
    alert(body.detail || text);
    return;
  }
  status.textContent = `Apply completed: ${body.reassigned_documents} documents reassigned, ${body.deleted_correspondents} correspondents deleted · manual boundary ${body.manual_review_boundary_id}`;
  output.textContent = JSON.stringify(body, null, 2);
  document.querySelector("#apply").disabled = true;
};
refresh();
