"""Execução de lotes com progresso visível, independente do formato da interface.

Cada combinação (mapa x território) passa por: descoberta de bandas, consulta
à biblioteca (pula o que já existe), geração no diretório temporário e
publicação na biblioteca local ou no Drive. Nada é sobrescrito.
"""
from datetime import datetime
from html import escape
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from time import monotonic

import ipywidgets as w

from .i18n import N_, _, ui_text
from .engine import discover, generate, validate
from .estimate import SECONDS_PER_COMBO, Calibration, human_size
from .layout import combination_path, ensure_unique_paths
from .library import assess, fingerprint, ingest, variant_for
from .progress import ProgressState

DRIVE_ROOT = Path("/content/drive/MyDrive")
STATE_LABELS = {"disponivel": N_("✓ Já existe"), "parcial": N_("◐ Incompleto (será completado)"),
                "nao_gerado": N_("○ Será gerado"), "verificar": N_("⚠ Verificar pasta")}
# Etapas que o pipeline informa; traduzidas na hora de mostrar.
STAGES = [N_("Preparando lote"), N_("Preparando combinação"), N_("Preparando download"), N_("Baixando frames"),
          N_("Consultando períodos e biblioteca"), N_("Preparando frames cartográficos"),
          N_("Preparando mapas com ano, escala e norte"), N_("Montando colagens"), N_("Gerando GIF principal"),
          N_("Gerando GIFs especiais"), N_("Finalizando frames e GIFs"), N_("Organizando arquivos selecionados"),
          N_("Enviando ao Drive"), N_("Publicando arquivos"), N_("Preparando")]


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def drive_mounted():
    return DRIVE_ROOT.is_dir()


def header_for(origin, layer, territory):
    from .legends import theme_label
    theme = ui_text(theme_label(layer["theme"], "pt")) if layer.get("theme") else ""
    collection = layer.get("collection") or ""
    collection = _("Coleção {c}", c=collection) if str(collection)[:1].isdigit() else collection
    parts = [ui_text(origin), theme, collection, ui_text(layer["name"]), territory["name"]]
    return " / ".join(str(p) for p in parts if p)


class BatchRunner:
    """Painel de progresso + execução em segundo plano de um lote."""

    def __init__(self, output_root, temp_root, project):
        self.output_root = Path(output_root)
        self.temp_root = Path(temp_root)
        self.project = project
        self.lock = Lock()
        self.on_finished = None  # callback(produced_files, summary_dict)
        self.on_combo = None     # callback(índice, estado, info): alimenta a vitrine
        self.on_start = None     # callback(combos): a vitrine cria um cartão por combinação
        self.workers = 1         # combinações geradas ao mesmo tempo
        self.calibration = Calibration()  # ritmo real desta sessão, para as próximas estimativas
        self.status = w.HTML(f"<b>{_('Pronto.')}</b>")
        self.overall = w.IntProgress(value=0, min=0, max=1, description=_("Lote:"), layout=w.Layout(width="95%"))
        self.sprout = w.HTML(
            value=('<style>@keyframes gif-factory-sprout {0%,100% { transform: rotate(-8deg); }'
                   ' 50% { transform: rotate(8deg); }} .gif-factory-sprout { display: inline-block;'
                   ' transform-origin: bottom center; animation: gif-factory-sprout 1.5s ease-in-out infinite; }'
                   '@media (prefers-reduced-motion: reduce) { .gif-factory-sprout { animation: none; }}</style>'
                   '<span role="status" aria-label="Processamento em andamento"><span class="gif-factory-sprout"'
                   ' aria-hidden="true">🌱</span> <span style="color:#4b7255; font-size:12px">' + _("Cultivando mapas…") + '</span></span>'),
            layout=w.Layout(display="none"))
        self.log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="220px", overflow="auto"))

    def widget(self):
        log = w.Accordion([self.log], selected_index=None)
        log.set_title(0, _("Registro detalhado"))
        return w.VBox([self.sprout, self.status, self.overall, log])

    def library_root(self, automatic, destination):
        if automatic:
            if not drive_mounted():
                raise RuntimeError(_("O Google Drive não está conectado; desmarque o Drive ou rode a célula 1 de novo."))
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
                result.append(((origin, layer, territory), _(STATE_LABELS[state])))
            except Exception as exc:
                result.append(((origin, layer, territory), f"⚠ {exc}"))
        return result

    def start(self, combos, kinds, automatic=False, destination=None, wait=True):
        """Inicia o lote. Retorna a thread, ou None se nada foi iniciado."""
        if not self.lock.acquire(blocking=False):
            self.status.value = f"<b>{_('Já existe um lote em andamento.')}</b>"
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
            self.status.value = f"<b>{_('Erro ao iniciar o lote:')}</b> {escape(str(exc))}"
            self.log.append_stdout(f"Erro ao iniciar lote: {exc}\n")
            return None

    def _launch(self, combos, kinds, automatic, destination):
        if not combos or not kinds:
            self.status.value = f"<b>{_('Escolha ao menos um mapa, um território e um tipo de arquivo.')}</b>"
            return None
        try:
            for _x, layer, territory in combos:
                validate([layer], [territory])
            ensure_unique_paths(combos)
        except ValueError as exc:
            self.status.value = f"<b>{_('Seleção inválida:')}</b> {escape(str(exc))}"
            return None
        self.sprout.layout.display = ""
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        self.status.value = f"<b>▶ {_('INICIANDO')}</b> — {_('preparando lote...')}"
        self.overall.value = 0
        self.overall.max = len(combos)
        self.log.clear_output()
        if self.on_start:
            self.on_start(combos)
        thread = Thread(target=self._work, args=(combos, kinds, automatic, destination, run_id), daemon=True)
        thread.start()
        return thread

    def _work(self, combos, kinds, automatic, destination, run_id):
        total = len(combos)
        workers = max(1, min(int(self.workers or 1), total))
        states = [ProgressState(1) for _x in combos]
        active = {}  # número -> cabeçalho, na ordem em que começaram
        stop_clock = Event()
        counts = {"ok": 0, "failed": 0, "skipped": 0, "done": 0, "saved": 0, "work": 0}
        count_lock = Lock()
        produced = []
        started = monotonic()

        def notify(number, state, **info):
            if self.on_combo:
                try:
                    self.on_combo(number - 1, state, info)
                except Exception as exc:
                    self.log.append_stdout(f"Aviso da vitrine: {exc}\n")

        def render():
            with count_lock:
                running = list(active.items())
                done = counts["done"]
            lines = []
            for number, header in running:
                snap = states[number - 1].snapshot()
                details = f" — {escape(snap['detail'])}" if snap["detail"] else ""
                eta = ""
                if snap["stage"] == "Baixando frames" and snap["eta"] is not None:
                    eta = " · " + _("faltam ~{tempo}", tempo=duration(snap["eta"]))
                lines.append(f"<br>▸ <b>{number}/{total}</b> {escape(header)}<br>&nbsp;&nbsp;&nbsp;"
                             f"{escape(_(snap['stage']))}{details} · {duration(snap['combo_elapsed'])}{eta}")
            together = " · " + _("{n} ao mesmo tempo", n=workers) if workers > 1 else ""
            progress = _("{feitas} de {total} combinações processadas", feitas=done, total=total)
            self.status.value = (f"<b>◉ {_('LOTE EM ANDAMENTO')}</b> — {progress}{together}"
                                 f"<br>{_('Tempo total:')} {duration(monotonic() - started)}" + "".join(lines))
            self.overall.value = done
            for number, _x in running:
                snap = states[number - 1].snapshot()
                notify(number, "gerando", detail=snap["stage"], done=snap["frame_done"], total=snap["frame_total"])

        def tick():
            while not stop_clock.wait(1):
                try:
                    render()
                except Exception as exc:
                    self.log.append_stdout(f"Aviso de atualização do painel: {exc}\n")

        def run_one(number, origin, layer, territory):
            header = header_for(origin, layer, territory)
            state = states[number - 1]
            state.begin(number, header)
            with count_lock:
                active[number] = header
            notify(number, "gerando", detail="Preparando")
            self.log.append_stdout(f"INICIANDO {number}/{total}: {header}\n")
            tag = f"[{number}/{total}] " if workers > 1 else ""

            def event(kind, *args):
                if kind == "discovered" and args[0]:
                    self.log.append_stdout(f"{tag}{len(args[0])} períodos: {args[0][0]} → {args[0][-1]}\n")
                if kind == "stage":
                    self.log.append_stdout(f"{tag}{args[0]}...\n")
                state.update(kind, *args)
                if kind in ("frame_retry", "frame_failed"):
                    self.log.append_stdout(tag + state.snapshot()["detail"] + "\n")
                render()

            result, files, folder, error = "falha", [], None, ""
            try:
                relative = combination_path(origin, layer, territory)
                local_root = self.output_root / relative
                primary_root = (Path(destination) / relative) if automatic else local_root
                folder = primary_root
                if automatic and not drive_mounted():
                    raise RuntimeError("Drive não conectado; desmarque o Drive ou conecte-o de novo.")
                state.update("stage", "Consultando períodos e biblioteca")
                render()
                all_bands, selected_bands = discover(layer)
                variant = variant_for(all_bands, selected_bands)
                signature = fingerprint(layer, territory, selected_bands)
                existing = assess(primary_root, variant, signature, kinds)
                if existing == "disponivel":
                    self.log.append_stdout(f"JÁ EXISTE {number}/{total}: não reprocessado ({primary_root}).\n")
                    files = artifacts_in(primary_root)
                    result = "existe"
                    return
                if existing == "verificar":
                    raise RuntimeError("Há arquivos incompatíveis ou sem metadados nesta pasta; nada foi sobrescrito.")
                if existing == "parcial":
                    self.log.append_stdout(f"{tag}Resultado incompleto: gerando apenas o que falta.\n")
                missing = {kind for kind in kinds if assess(primary_root, variant, signature, {kind}) != "disponivel"}
                run_root = self.temp_root / relative / variant / run_id
                stage = run_root / "entrega"
                configured = dict(layer, bands=selected_bands)
                generated = generate(configured, territory, stage, missing, None, project=self.project,
                                     on_progress=event, work_dir=run_root / "processamento", variant=variant)
                state.update("stage", "Enviando ao Drive" if automatic else "Publicando arquivos")
                render()
                ingest(stage, primary_root, variant, signature, selected_bands, layer, territory)
                if automatic:
                    try:
                        ingest(stage, local_root, variant, signature, selected_bands, layer, territory)
                    except Exception as exc:
                        self.log.append_stdout(f"{tag}Aviso: cópia local não atualizada: {exc}\n")
                files = artifacts_in(primary_root)
                result = "pronto"
                snap = state.snapshot()
                self.calibration.add(snap["combo_elapsed"] - SECONDS_PER_COMBO, snap["frame_total"])
                with count_lock:
                    counts["saved"] += sum(size_of(f) for f in generated)
                    counts["work"] += sum(size_of(f) for f in run_root.rglob("*"))
                self.log.append_stdout(f"FINALIZADO {number}/{total}: {len(generated)} arquivos em {primary_root}\n")
            except Exception as exc:
                error = str(exc)
                self.log.append_stdout(f"FALHA {number}/{total}: {exc}\n")
            finally:
                with count_lock:
                    active.pop(number, None)
                    counts["done"] += 1
                    counts[{"pronto": "ok", "existe": "skipped", "falha": "failed"}[result]] += 1
                    produced.extend(files)
                state.finish_combo()
                notify(number, result, detail=error, files=files, folder=folder)
                render()

        ticker = Thread(target=tick, daemon=True)
        try:
            ticker.start()
            if workers == 1:
                for number, (origin, layer, territory) in enumerate(combos, 1):
                    run_one(number, origin, layer, territory)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gif-factory") as pool:
                    futures = [pool.submit(run_one, number, *combo) for number, combo in enumerate(combos, 1)]
                    for future in futures:
                        future.result()
            stop_clock.set()
            if ticker.is_alive():
                ticker.join()
            ok, failed, skipped = counts["ok"], counts["failed"], counts["skipped"]
            title = (_("✓ LOTE CONCLUÍDO") if not failed else
                     _("⚠ CONCLUÍDO COM FALHAS") if ok or skipped else _("✕ LOTE FALHOU"))
            where = (_("no Google Drive") if automatic else _("nesta sessão do Colab")) if (ok or skipped) else "—"
            self.overall.value = total
            counts_line = _("{ok} gerado(s), {existentes} já disponível(is), {falhas} falha(s)",
                            ok=ok, existentes=skipped, falhas=failed)
            self.status.value = (f"<b>{title}</b> · {counts_line}"
                                 f"<br>{_('Arquivos:')} {where}<br>{_('Tempo total:')} {duration(monotonic() - started)}")
            if counts["saved"]:
                copies = " " + _("(mais a cópia nesta sessão)") if automatic else ""
                volume = _("Volume: {salvos} salvos{copias} · {trabalho} de arquivos de trabalho em {pasta}",
                           salvos=human_size(counts["saved"] / 2 ** 20), copias=copies,
                           trabalho=human_size(counts["work"] / 2 ** 20), pasta=escape(str(self.temp_root)))
                self.status.value += f"<br>{volume}"
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


def size_of(path):
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def artifacts_in(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in {".gif", ".png"})
