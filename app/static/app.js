const form        = document.getElementById("form");
const results     = document.getElementById("results");
const nomInput    = document.getElementById("nom");
const prenomInput = document.getElementById("prenom");

let state = { nom: [], prenom: [] };

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const nom    = nomInput.value.trim();
  const prenom = prenomInput.value.trim();
  if (!nom && !prenom) return;
  await doSearch(nom, prenom);
});

async function doSearch(nom, prenom) {
  setLoading();
  const params = new URLSearchParams();
  if (nom)    params.set("nom", nom);
  if (prenom) params.set("prenom", prenom);

  try {
    const res = await fetch(`/api/search?${params}`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    state = { nom: data.nom || [], prenom: data.prenom || [] };
    render(nom, prenom);
  } catch {
    results.hidden = false;
    results.innerHTML = `<p class="error">Une erreur est survenue. Veuillez réessayer.</p>`;
  }
}

function setLoading() {
  results.hidden = false;
  results.innerHTML = `<p class="loading">Recherche en cours...</p>`;
}

function render(nom, prenom) {
  results.innerHTML = "";

  if (state.prenom.length) {
    results.appendChild(renderBlock("Prénom", state.prenom, "prenom", prenom));
  }
  if (state.nom.length) {
    results.appendChild(renderBlock("Nom de famille", state.nom, "nom", nom));
  }
  if (!state.nom.length && !state.prenom.length) {
    results.innerHTML = `<p class="empty">Aucun résultat trouvé.</p>`;
  }
}

function renderBlock(label, candidates, type, query) {
  const top  = candidates[0];
  const rest = candidates.slice(1).filter(c => c.score > 0.05);

  const block = document.createElement("div");
  block.className = "block";

  // Badge exact / suggestion
  const isApprox  = top.match_type === "approximate";
  const badgeHtml = isApprox ? `<span class="match-badge">suggestion</span>` : "";

  // Famille linguistique
  const langHtml = top.language_family
    ? `<span class="lang-tag">${escHtml(top.language_family)}</span>`
    : "";

  // Texte d'origine
  const originHtml = top.origin
    ? `<p class="origin-text">${escHtml(top.origin)}</p>`
    : `<p class="empty">Origine non disponible pour ce nom.</p>`;

  // Variantes orthographiques
  let variantsHtml = "";
  if (top.variants && top.variants.length) {
    const items = top.variants.map(v => `<span class="variant-tag">${escHtml(v)}</span>`).join("");
    variantsHtml = `
      <div class="meta-block">
        <p class="meta-label">Variantes orthographiques</p>
        <div class="tag-list">${items}</div>
      </div>`;
  }

  // Popularité INSEE (prénoms uniquement)
  let inseeHtml = "";
  if (top.insee) {
    const { count, rank } = top.insee;
    const countFmt = count.toLocaleString("fr-FR");
    const rankLabel = rank === 1 ? "1er" : `${rank.toLocaleString("fr-FR")}e`;
    const typeLabel = type === "prenom" ? "prénom le plus donné" : "nom de famille le plus porté";
    inseeHtml = `
      <div class="meta-block">
        <p class="meta-label">Popularité en France (INSEE)</p>
        <p class="insee-stat">${rankLabel} ${typeLabel} · <span class="insee-count">${countFmt}</span> porteurs enregistrés</p>
      </div>`;
  }

  // Voir aussi
  let seeAlsoHtml = "";
  if (top.see_also && top.see_also.length) {
    const links = top.see_also.map(n =>
      `<button class="see-also-btn" data-name="${escAttr(n)}" data-type="${type}">${escHtml(capitalize(n))}</button>`
    ).join("");
    seeAlsoHtml = `
      <div class="meta-block">
        <p class="meta-label">Voir aussi</p>
        <div class="tag-list">${links}</div>
      </div>`;
  }

  // Autres correspondances (si résultat approché)
  let suggestionsHtml = "";
  if (isApprox && rest.length) {
    const btns = rest.map(c =>
      `<button class="suggestion-btn" data-type="${type}" data-name="${escAttr(c.name)}">${escHtml(capitalize(c.name))}</button>`
    ).join("");
    suggestionsHtml = `
      <div class="meta-block">
        <p class="meta-label">Autres correspondances</p>
        <div class="tag-list">${btns}</div>
      </div>`;
  }

  block.innerHTML = `
    <p class="block-label">${label}</p>
    <div class="block-header">
      <p class="block-name">${escHtml(capitalize(top.name))}${badgeHtml}</p>
      ${langHtml}
    </div>
    ${originHtml}
    ${inseeHtml}
    ${variantsHtml}
    ${seeAlsoHtml}
    ${suggestionsHtml}
  `;

  // Clic sur "Autres correspondances" → swap
  block.querySelectorAll(".suggestion-btn").forEach(btn => {
    btn.addEventListener("click", () => swapCandidate(btn, block, label, type, query));
  });

  // Clic sur "Voir aussi" → nouvelle recherche du nom
  block.querySelectorAll(".see-also-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      if (type === "nom") {
        nomInput.value = name;
        doSearch(name, prenomInput.value.trim());
      } else {
        prenomInput.value = name;
        doSearch(nomInput.value.trim(), name);
      }
    });
  });

  return block;
}

function swapCandidate(btn, block, label, type, query) {
  const name = btn.dataset.name;
  const list = state[type];
  const idx  = list.findIndex(c => c.name === name);
  if (idx < 0) return;
  const [item] = list.splice(idx, 1);
  list.unshift(item);
  block.replaceWith(renderBlock(label, list, type, query));
}

function capitalize(str) {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escAttr(str) {
  return String(str).replace(/"/g, "&quot;");
}
