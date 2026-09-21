/* ChatJUCHLM — cliente orquestado (§24-25). Botones dinámicos desde Capability Registry. */
(function () {
  const store = { session_id: localStorage.getItem("juchlm_sid") || null };

  async function postMessage(message, action_id) {
    const res = await fetch("/api/conversation/message", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: store.session_id, message: message || "", action_id: action_id || null }),
    });
    if (!res.ok) throw new Error("Error de red");
    const data = await res.json();
    if (data.session_id) { store.session_id = data.session_id; localStorage.setItem("juchlm_sid", data.session_id); }
    return data;
  }

  async function loadCapabilities() {
    try {
      const r = await fetch("/api/capabilities");
      return await r.json();
    } catch { return { capabilities: [] }; }
  }

  function renderActions(container, actions, onPick) {
    container.innerHTML = "";
    (actions || []).forEach((a) => {
      const b = document.createElement("button");
      b.className = "juchlm-action-btn";
      b.textContent = a.label;
      b.onclick = () => onPick(a);
      if (a.type === "download" && a.id === "descargar_constancia_usuario") {
        b.onclick = () => { window.location.href = "/api/tramites/constancia-usuario"; };
      }
      container.appendChild(b);
    });
  }

  window.JUCHLM = { postMessage, loadCapabilities, renderActions, store };
})();
