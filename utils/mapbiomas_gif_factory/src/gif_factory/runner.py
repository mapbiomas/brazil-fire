"""Execução de lotes com progresso visível, independente do formato da interface.

Cada combinação (mapa x território) passa por: descoberta de bandas, consulta
à biblioteca (pula o que já existe), geração no diretório temporário e
publicação na biblioteca local ou no Drive. Nada é sobrescrito.
"""
from datetime import datetime
from html import escape
from pathlib import Path
from threading import Event, Lock, Thread

import ipywidgets as w

from .engine import discover, generate, validate
from .layout import combination_path, ensure_unique_paths
from .library import assess, fingerprint, ingest, variant_for
from .progress import ProgressState

DRIVE_ROOT = Path("/content/drive/MyDrive")
STATE_LABELS = {"disponivel": "✓ Já existe", "parcial": "◐ Incompleto (será completado)",
                "nao_gerado": "○ Será gerado", "verificar": "⚠ Verificar pasta"}


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def drive_mounted():
    return DRIVE_ROOT.is_dir()


def header_for(origin, layer, territory):
    from .legends import theme_label
    theme = theme_label(layer["theme"], "pt") if layer.get("theme") else ""
    collection = layer.get("collection") or ""
    collection = f"Coleção {collection}" if str(collection)[:1].isdigit() else collection
    parts = [origin, theme, collection, layer["name"], territory["name"]]
    return " / ".join(str(p) for p in parts if p)


class BatchRunner:
    """Painel de progresso + execução em segundo plano de um lote."""

    def __init__(self, output_root, temp_root, project):
        self.output_root = Path(output_root)
        self.temp_root = Path(temp_root)
        self.project = project
        self.lock = Lock()
        self.on_finished = None  # callback(produced_files, summary_dict)
        self.status = w.HTML("<b>Pronto.</b>")
        self.overall = w.IntProgress(value=0, min=0, max=1, description="Lote:", layout=w.Layout(width="95%"))
        self.progress = w.IntProgress(value=0, min=0, max=1, description="Download:",
                                      layout=w.Layout(width="95%", display="none"))
        self.sprout = w.HTML(
            value=('<style>@keyframes gif-factory-sprout {0%,100% { transform: rotate(-8deg); }'
                   ' 50% { transform: rotate(8deg); }} .gif-factory-sprout { display: inline-block;'
                   ' transform-origin: bottom center; animation: gif-factory-sprout 1.5s ease-in-out infinite; }'
                   '@media (prefers-reduced-motion: reduce) { .gif-factory-sprout { animation: none; }}</style>'
                   '<span role="status" aria-label="Processamento em andamento"><span class="gif-factory-sprout"'
                   ' aria-hidden="true">🌱</span> <span style="color:#4b7255; font-size:12px">Cultivando mapas…</span></span>'),
            layout=w.Layout(display="none"))
        self.log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="220px", overflow="auto"))

    def widget(self):
        return w.VBox([self.sprout, self.status, self.overall, self.progress, self.log])

    def library_root(self, automatic, destination):
        if automatic:
            if not drive_mounted():
                raise RuntimeError("O Google Drive não está conectado; desmarque o Drive ou rode a célula 1 de novo.")
            return Path(destination)
        return self.output_root

    def availability(self, combos, kinds, automatic, destination):
        """[(combinação, estado ou mensagem de erro)] sem gerar nada."""
        root = self.library_root(automatic, destination)
        result = []
        for origin, layer, territory in combos:
            try:
                all_bands, bands = discover(layer)
                variant = variant_for(all_bands, bands)
                state = assess(root / combination_path(origin, layer, territory), variant,
                               fingerprint(layer, territory, bands), set(kinds))
                result.append(((origin, layer, territory), STATE_LABELS[state]))
            except Exception as exc:
                result.append(((origin, layer, territory), f"⚠ {exc}"))
        return result

    def start(self, combos, kinds, automatic=False, destination=None, wait=True):
        """Inicia o lote. Retorna a thread, ou None se nada foi iniciado."""
        if not self.lock.acquire(blocking=False):
            self.status.value = "<b>Já existe um lote em andamento.</b>"
            return None
        thread = None
        try:
            thread = self._launch(combos, set(kinds), automatic, destination)
            if thread is None:
                self.lock.release()
                return None
            if wait:
                thread.join()
            return thread
        except Exception as exc:
            if thread is None:
                self.lock.release()
            self.sprout.layout.display = "none"
            self.status.value = f"<b>Erro ao iniciar o lote:</b> {escape(str(exc))}"
            self.log.append_stdout(f"Erro ao iniciar lote: {exc}\n")
            return None

    def _launch(self, combos, kinds, automatic, destination):
        if not combos or not kinds:
            self.status.value = "<b>Escolha ao menos um mapa, um território e um tipo de arquivo.</b>"
            return None
        try:
            for _, layer, territory in combos:
                validate([layer], [territory])
            ensure_unique_paths(combos)
        except ValueError as exc:
            self.status.value = f"<b>Seleção inválida:</b> {escape(str(exc))}"
            return None
        self.sprout.layout.display = ""
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        self.status.value = "<b>▶ INICIANDO</b> — preparando lote..."
        self.progress.value = 0
        self.overall.value = 0
        self.overall.max = len(combos)
        self.log.clear_output()
        thread = Thread(target=self._work, args=(combos, kinds, automatic, destination, run_id), daemon=True)
        thread.start()
        return thread

    def _work(self, combos, kinds, automatic, destination, run_id):
        total = len(combos)
        state = ProgressState(total)
        stop_clock = Event()
        ok = failed = skipped = 0
        produced = []

        def render():
            snap = state.snapshot()
            eta = "calculando…" if snap["eta"] is None else "~" + duration(snap["eta"])
            details = f"<br>{escape(snap['detail'])}" if snap["detail"] else ""
            download = snap["stage"] == "Baixando frames"
            remaining = f"<br>Previsão do download atual: {eta}" if download else ""
            self.status.value = (f"<b>◉ LOTE EM ANDAMENTO</b> — {snap['completed']} de {snap['total']} combinações processadas"
                                 f"<br>{escape(snap['header'])}<br><b>{escape(snap['stage'])}</b>{details}"
                                 f"<br>Tempo total: {duration(snap['elapsed'])} · combinação: {duration(snap['combo_elapsed'])}{remaining}")
            self.overall.value = snap["completed"]
            self.progress.layout.display = "" if download else "none"
            if download:
                self.progress.max = max(1, snap["frame_total"])
                self.progress.value = min(snap["frame_done"], self.progress.max)

        def tick():
            while not stop_clock.wait(1):
                try:
                    render()
                except Exception as exc:
                    self.log.append_stdout(f"Aviso de atualização do painel: {exc}\n")

        ticker = Thread(target=tick, daemon=True)
        try:
            ticker.start()
            for number, (origin, layer, territory) in enumerate(combos, 1):
                header = header_for(origin, layer, territory)
                state.begin(number, header)
                render()
                self.log.append_stdout(f"INICIANDO {number}/{total}: {header}\n")

                def event(kind, *args):
                    if kind == "discovered" and args[0]:
                        self.log.append_stdout(f"{len(args[0])} períodos: {args[0][0]} → {args[0][-1]}\n")
                    if kind == "stage":
                        self.log.append_stdout(f"{args[0]}...\n")
                    state.update(kind, *args)
                    if kind in ("frame_retry", "frame_failed"):
                        self.log.append_stdout(state.snapshot()["detail"] + "\n")
                    render()

                try:
                    relative = combination_path(origin, layer, territory)
                    local_root = self.output_root / relative
                    primary_root = (Path(destination) / relative) if automatic else local_root
                    if automatic and not drive_mounted():
                        raise RuntimeError("Drive não conectado; desmarque o Drive ou conecte-o de novo.")
                    state.update("stage", "Consultando períodos e biblioteca")
                    render()
                    all_bands, selected_bands = discover(layer)
                    variant = variant_for(all_bands, selected_bands)
                    signature = fingerprint(layer, territory, selected_bands)
                    existing = assess(primary_root, variant, signature, kinds)
                    if existing == "disponivel":
                        skipped += 1
                        self.log.append_stdout(f"JÁ EXISTE {number}/{total}: não reprocessado ({primary_root}).\n")
                        produced.extend(sorted(p for p in primary_root.rglob("*") if p.suffix.lower() in {".gif", ".png"}))
                        continue
                    if existing == "verificar":
                        raise RuntimeError("Há arquivos incompatíveis ou sem metadados nesta pasta; nada foi sobrescrito.")
                    if existing == "parcial":
                        self.log.append_stdout("Resultado incompleto: gerando apenas o que falta.\n")
                    missing = {kind for kind in kinds if assess(primary_root, variant, signature, {kind}) != "disponivel"}
                    run_root = self.temp_root / relative / variant / run_id
                    stage = run_root / "entrega"
                    configured = dict(layer, bands=selected_bands)
                    files = generate(configured, territory, stage, missing, None, project=self.project,
                                     on_progress=event, work_dir=run_root / "processamento", variant=variant)
                    state.update("stage", "Enviando ao Drive" if automatic else "Publicando arquivos")
                    render()
                    ingest(stage, primary_root, variant, signature, selected_bands, layer, territory)
                    if automatic:
                        try:
                            ingest(stage, local_root, variant, signature, selected_bands, layer, territory)
                        except Exception as exc:
                            self.log.append_stdout(f"Aviso: cópia local não atualizada: {exc}\n")
                    produced.extend(primary_root / Path(f).relative_to(stage) for f in files)
                    ok += 1
                    self.log.append_stdout(f"FINALIZADO {number}/{total}: {len(files)} arquivos em {primary_root}\n")
                except Exception as exc:
                    failed += 1
                    self.log.append_stdout(f"FALHA {number}/{total}: {exc}\n")
                finally:
                    state.finish_combo()
                    render()
            stop_clock.set()
            if ticker.is_alive():
                ticker.join()
            title = ("✓ LOTE CONCLUÍDO" if not failed else
                     "⚠ CONCLUÍDO COM FALHAS" if ok or skipped else "✕ LOTE FALHOU")
            where = ("no Google Drive" if automatic else "nesta sessão do Colab") if (ok or skipped) else "—"
            self.status.value = (f"<b>{title}</b> · {ok} gerado(s), {skipped} já disponível(is), {failed} falha(s)"
                                 f"<br>Arquivos: {where}<br>Tempo total: {duration(state.snapshot()['elapsed'])}")
            if self.on_finished:
                try:
                    self.on_finished(produced, {"ok": ok, "skipped": skipped, "failed": failed})
                except Exception as exc:
                    self.log.append_stdout(f"Aviso ao listar resultados: {exc}\n")
        except Exception as exc:
            self.status.value = f"<b>Erro ao finalizar o lote:</b> {escape(str(exc))}"
            self.log.append_stdout(f"Erro ao finalizar lote: {exc}\n")
        finally:
            stop_clock.set()
            if ticker.is_alive():
                ticker.join()
            self.sprout.layout.display = "none"
            self.lock.release()
