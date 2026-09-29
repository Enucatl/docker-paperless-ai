// Run the shipped script against a minimal DOM; HTML parsing is forbidden.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const plan = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
  }
  set innerHTML(value) {
    assert.fail(`Unsafe HTML insertion: ${value}`);
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.children = children;
  }
}
const elements = new Map();
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, new Element(selector));
    return elements.get(selector);
  },
  createElement(tag) {
    return new Element(tag);
  },
};
const submissions = [];
let reviewedPlanRequests = 0;
const fetch = async (url, options) => {
  if (url === "api/decisions") {
    const decision = JSON.parse(options.body);
    submissions.push(decision);
    plan.review_decisions = [decision];
  }
  if (url === "api/reviewed-plan") reviewedPlanRequests++;
  return { ok: true, json: async () => (url === "api/plan" ? plan : {}) };
};
const context = vm.createContext({ document, fetch, alert: assert.fail });
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
(async () => {
  await vm.runInContext("refresh()", context);
  let card = elements.get("#pairs").children[0];
  const pair = plan.review_candidates[0];
  assert.equal(card.children[0].textContent, pair.left_name);
  assert.equal(card.children[2].textContent, pair.right_name);
  assert.equal(
    card.children[4].textContent,
    `${pair.reason} · candidate score —`,
  );
  assert.deepEqual(
    card.children.filter((x) => x instanceof Element).map((x) => x.tag),
    ["strong", "strong", "p", "button", "button"],
  );
  const beforeDecision = reviewedPlanRequests;
  await card.children[5].onclick();
  assert.equal(reviewedPlanRequests, beforeDecision + 1);
  assert.deepEqual(submissions, [
    { pair_key: "2:10|3:20", decision: "approve" },
  ]);
  card = elements.get("#pairs").children[0];
  assert.equal(card.children[5].textContent, "Approved");
  assert.equal(card.children[5].className, "approved");
  await card.children[6].onclick();
  assert.equal(submissions[1].pair_key, "2:10|3:20");
  assert.equal(
    elements.get("#pairs").children[0].children[6].textContent,
    "Rejected",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
