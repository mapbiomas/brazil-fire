import os
import json
import numpy as np
from datetime import datetime
TF_AVAILABLE = None
TF_ERROR = None
from M0_auth_config import CONFIG, GLOBAL_OPTS, model_path, gcs_samples_path
from M_cache import _get_fs

from M4_data_extractor import compute_normalizer, normalize
from M_dnn import forward_embeddings, forward_predict


def _get_tf(force=False):
    """Carrega o TensorFlow apenas quando necessário (Lazy Load).
    
    Args:
        force: Se True, ignora o cache e tenta carregar novamente.
    """
    global TF_AVAILABLE, TF_ERROR
    if not force and TF_AVAILABLE is True:
        return tf
    if not force and TF_AVAILABLE is False:
        return None
    
    try:
        import tensorflow.compat.v1 as _tf
        try:
            _tf.compat.v1.disable_v2_behavior()
        except Exception:
            pass
        globals()['tf'] = _tf
        TF_AVAILABLE = True
        return _tf
    except Exception as e:
        TF_AVAILABLE = False
        TF_ERROR = str(e)
        return None

class ModelTrainer:
    def __init__(self, num_input, layers=None, lr=None, seed=42):
        _get_tf()
        self.num_input  = num_input
        self.layers = layers or CONFIG.get('model_layers', [64, 32])
        self.lr     = lr     or CONFIG.get('model_lr', 0.001)
        self.seed       = seed
        self.graph      = None
        self.session    = None
        self.norm_stats = None

    def _build_graph(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        self.x   = tf.placeholder(tf.float32, [None, self.num_input], name='x')
        self.y   = tf.placeholder(tf.float32, [None, 1],              name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        layer = self.x
        for i, n_units in enumerate(self.layers):
            layer = tf.keras.layers.Dense(
                n_units,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                name=f'fc_{i}'
            )(layer)
            layer = tf.nn.dropout(layer, rate=1 - self.keep_prob)
        
        self.latent_tensor = layer # Captura para visualização ao vivo
        self.logits = tf.keras.layers.Dense(1, name='output')(layer)
        self.pred   = tf.nn.sigmoid(self.logits)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        )
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver    = tf.train.Saver()

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=None, n_iters=None, keep_prob=0.8, logger=None, update_chart_fn=None, snapshot_dir=None):
        self._X_raw = X_train
        self._y_raw = y_train
        
        batch_size = batch_size or CONFIG.get('model_batch', 1000)
        n_iters    = n_iters    or CONFIG.get('model_iters', 5000)
        self._batch_size = batch_size
        self._n_iters = n_iters
        self._keep_prob = keep_prob
        self._split_ratio = 0.2

        # Normalização baseada SEMPRE no conjunto de treino para evitar data leakage
        self.norm_stats = compute_normalizer(X_train)
        X_tr = normalize(X_train, self.norm_stats)
        y_tr = y_train.reshape(-1, 1)

        if X_val is not None and y_val is not None:
            X_te = normalize(X_val, self.norm_stats)
            y_te = y_val.reshape(-1, 1)
        else:
            # Fallback para split interno se não for provido
            n = len(X_tr)
            idx = np.random.permutation(n)
            n_split = int(n * 0.8)
            X_te = X_tr[idx[n_split:]]
            y_te = y_tr[idx[n_split:]]
            X_tr = X_tr[idx[:n_split]]
            y_tr = y_tr[idx[:n_split]]

        n_train = len(X_tr)

        # Amostra fixa para visualização ao vivo (Playground Style)
        viz_idx = np.random.choice(len(X_te), min(500, len(X_te)), replace=False)
        X_viz, y_viz = X_te[viz_idx], y_te[viz_idx]

        self._build_graph()
        history = {'loss': [], 'acc': [], 'val_acc': [], 'steps': []}
        # Data accumulation for time-travel
        snapshots_data = {'y_true': y_viz.flatten()}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.session = sess

            for i in range(1, n_iters + 1):
                b_idx = np.random.randint(0, n_train, batch_size)
                _, loss_val = sess.run(
                    [self.train_op, self.loss],
                    feed_dict={self.x: X_tr[b_idx], self.y: y_tr[b_idx], self.keep_prob: keep_prob}
                )

                if i % max(1, n_iters // 20) == 0 or i == n_iters:
                    pred_tr = sess.run(self.pred, feed_dict={self.x: X_tr, self.y: y_tr, self.keep_prob: 1.0})
                    acc_tr = ((pred_tr > 0.5).astype(int) == y_tr.astype(int)).mean()

                    pred_te = sess.run(self.pred, feed_dict={self.x: X_te, self.y: y_te, self.keep_prob: 1.0})
                    acc_te = ((pred_te > 0.5).astype(int) == y_te.astype(int)).mean()

                    history['steps'].append(i)
                    history['loss'].append(float(loss_val))
                    history['acc'].append(float(acc_tr))
                    history['val_acc'].append(float(acc_te))

                    # Extração de dados para o Playground Live
                    embeds, preds_viz = sess.run(
                        [self.latent_tensor, self.pred], 
                        feed_dict={self.x: X_viz, self.keep_prob: 1.0}
                    )
                    
                    # Accumulate for time-travel
                    step_idx = len(history['steps']) - 1
                    snapshots_data[f'embeds_{step_idx}'] = embeds
                    snapshots_data[f'preds_{step_idx}'] = preds_viz.flatten()

                    if logger:
                        logger(f"Iter {i:5d}/{n_iters} | Loss: {loss_val:.4f} | Acc Treino: {acc_tr:.3f} | Validação: {acc_te:.3f}", "info")
                    
                    if update_chart_fn:
                        update_chart_fn(history, embeds, preds_viz.flatten(), y_viz)

            self._saved_vars = {v.name: sess.run(v) for v in tf.global_variables()}
            self._history = history
            self.snapshot_dir = snapshot_dir
            
            # Salva o arquivo de dados de snapshots se o diretório existir
            if snapshot_dir:
                np.savez_compressed(os.path.join(snapshot_dir, "snapshots_data.npz"), **snapshots_data)

        return history

    def evaluate(self):
        from sklearn.metrics import confusion_matrix, classification_report
        
        X_norm = normalize(self._X_raw, self.norm_stats)
        layer = X_norm
        for i in range(len(self.layers)):
            W = self._saved_vars[f'fc_{i}/kernel:0']
            b = self._saved_vars[f'fc_{i}/bias:0']
            layer = np.maximum(0, np.dot(layer, W) + b)
            
        W_out = self._saved_vars['output/kernel:0']
        b_out = self._saved_vars['output/bias:0']
        logits = np.dot(layer, W_out) + b_out
        # Sigmoid robusta para evitar overflow
        preds = (1 / (1 + np.exp(-np.clip(logits, -20, 20)))).flatten() > 0.5
        
        cm = confusion_matrix(self._y_raw, preds)
        rep = classification_report(self._y_raw, preds, output_dict=True, zero_division=0)
        return cm, rep

    def load(self, training_id, shortname):
        import tempfile
        from M0_auth_config import GLOBAL_OPTS, model_path
        base_path = model_path(training_id, shortname)
        fs = _get_fs()

        with tempfile.TemporaryDirectory() as tmpdir:
            for fname in ['weights.npz', 'metadata.json']:
                src = f"{CONFIG['bucket']}/{base_path}/{fname}"
                dest = os.path.join(tmpdir, fname)
                fs.get(src, dest)
            
            with open(os.path.join(tmpdir, 'metadata.json')) as f:
                hp = json.load(f)
                
            self.num_input   = hp['num_input']
            self.layers      = hp['layers']
            self.lr          = hp['lr']
            self.seed        = hp.get('seed', 42)
            self._bands_input = hp['bands_input']
            self.norm_stats  = {int(k): tuple(v) for k, v in hp['norm_stats'].items()}
            
            self._saved_vars = dict(np.load(os.path.join(tmpdir, 'weights.npz')))
            
    def predict(self, X_raw):
        """Aplica a inferência (forward pass manual usando os pesos numpy)."""
        if not hasattr(self, '_saved_vars'):
            raise RuntimeError("Modelo não treinado/carregado.")
        return forward_predict(
            self._saved_vars, self.layers, normalize(X_raw, self.norm_stats)
        )

    def get_embeddings(self, X_raw):
        """Extrai os embeddings (penúltima camada) usando os pesos numpy."""
        if not hasattr(self, '_saved_vars'):
            raise RuntimeError("Modelo não treinado/carregado.")
        return forward_embeddings(
            self._saved_vars, self.layers, normalize(X_raw, self.norm_stats)
        )

    @staticmethod
    def delete_model(training_id, shortname):
        """Deleta recursivamente a pasta do modelo no GCS."""
        from M_gcs import exists, rm
        from M0_auth_config import CONFIG
        base_uri = model_path(training_id, shortname)
        full_path = f"{CONFIG['bucket']}/{base_uri}"
        
        print(f" Deleting model folder (GCS): {full_path}")
        try:
            if exists(full_path):
                rm(full_path, recursive=True)
            return True
        except Exception as e:
            raise RuntimeError(f"No se pudo eliminar de GCS: {e}")

    def save_projector_files(self, training_id, shortname, X, y, logger=None):
        """Gera e salva arquivos .tsv para o TensorBoard Projector no GCS."""
        import tempfile, subprocess
        from M_gcs import upload
        from M0_auth_config import GLOBAL_OPTS
        base_path = model_path(training_id, shortname)
        
        if logger: logger("Generando embeddings para visualización (Projector)...", "info")
        embeddings = self.get_embeddings(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Vectors file (.tsv)
            vec_file = os.path.join(tmpdir, 'vectors.tsv')
            np.savetxt(vec_file, embeddings, delimiter='\t', fmt='%.6f')
            
            # 2. Metadata file (.tsv)
            meta_file = os.path.join(tmpdir, 'metadata.tsv')
            with open(meta_file, 'w', encoding='utf-8') as f:
                f.write("Index\tClass\tLabel\n")
                for i, val in enumerate(y):
                    label = "Fuego" if val == 1 else "No-Fuego"
                    f.write(f"{i}\t{val}\t{label}\n")
            
            # Upload to GCS
            dest_dir = f"{base_path}/projector"
            for fname in ['vectors.tsv', 'metadata.tsv']:
                src = os.path.join(tmpdir, fname)
                dest = f"{CONFIG['bucket']}/{dest_dir}/{fname}"
                if logger: logger(f"  [Enviar] Subiendo {fname} a GCS...", "info")
                upload(src, dest)
            
            if logger: 
                logger(f"[OK] Arquivos do Projector salvos em: {dest_dir}", "info")
                logger(f"[Dica] Dica: Baixe-os e suba em https://projector.tensorflow.org/", "info")

    def save(self, training_id, shortname, comment="", logger=None):
        import subprocess, tempfile
        from M_gcs import upload, copy
        from M0_auth_config import GLOBAL_OPTS
        base_path = model_path(training_id, shortname)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Weights
            np.savez(os.path.join(tmpdir, 'weights.npz'), **self._saved_vars)

            # 2. Metadata
            hp = {
                # Versão
                'metadata_version': 2,
                'training_date': datetime.now().isoformat(),

                # Identificação
                'training_id':  training_id,
                'shortname':    shortname,
                'country':      CONFIG['country'],
                'sensor':       ', '.join(GLOBAL_OPTS['SENSOR']) if isinstance(GLOBAL_OPTS['SENSOR'], list) else GLOBAL_OPTS['SENSOR'],
                'comment':      comment,

                # Dados de entrada
                'bands_input':  getattr(self, '_bands_input', CONFIG['bands_model_default']),
                'band_order':   sorted(getattr(self, '_bands_config', {}).keys()),
                'bands_config': getattr(self, '_bands_config', {}),
                'sample_collections': getattr(self, '_sample_collections', []),
                'sample_count': getattr(self, '_sample_count', {}),
                'num_input':    self.num_input,

                # Hiperparâmetros da rede
                'layers':       self.layers,
                'lr':           self.lr,
                'seed':         self.seed,
                'optimizer':    'Adam',

                # Hiperparâmetros de treino
                'n_iters':      self._n_iters,
                'batch_size':   self._batch_size,
                'keep_prob':    self._keep_prob,
                'split_ratio':  self._split_ratio,

                # Resultados
                'norm_stats':   {str(k): list(v) for k, v in self.norm_stats.items()},
                'history':      self._history,
                'rating':       0,

                # Snapshot do contexto global
                'global_opts': {
                    'sensor': GLOBAL_OPTS['SENSOR'],
                    'periodicity': GLOBAL_OPTS['PERIODICITY'],
                    'mosaic_methods': CONFIG['mosaic_methods'],
                    'sampling_campaign': CONFIG.get('campaign', 'MONITOR_01'),
                    'language': GLOBAL_OPTS.get('LANGUAGE', 'en'),
                },
            }
            with open(os.path.join(tmpdir, 'metadata.json'), 'w') as f:
                json.dump(hp, f, indent=2)

            for fname in ['weights.npz', 'metadata.json']:
                src  = os.path.join(tmpdir, fname)
                dest = f"{CONFIG['bucket']}/{base_path}/{fname}"
                upload(src, dest)
            
            # 2.5 Metrics
            if logger: logger("Generación y guardado de métricas de evaluación...", "info")
            cm, rep = self.evaluate()
            
            # --- NOTA IA AUTOMÁTICA (Baseada em regras técnicas) ---
            acc = rep.get('accuracy', 0)
            f1_fire = rep.get('1', {}).get('f1-score', 0)
            auto_rating = 1
            if f1_fire > 0.90 and acc > 0.95: auto_rating = 5
            elif f1_fire > 0.80 and acc > 0.90: auto_rating = 4
            elif f1_fire > 0.70 and acc > 0.85: auto_rating = 3
            elif f1_fire > 0.50: auto_rating = 2

            # --- SNAPSHOT DE DIAGNÓSTICO E t-SNE PARA CARGA INSTANTÂNEA ---
            try:
                from sklearn.decomposition import PCA
                idx_snap = np.random.choice(len(self._X_raw), min(800, len(self._X_raw)), replace=False)
                emb_snap = self.get_embeddings(self._X_raw[idx_snap])
                prd_snap = self.predict(self._X_raw[idx_snap])
                pca_snap = PCA(n_components=3); coords_pca = pca_snap.fit_transform(emb_snap)
                diag_snapshot = {
                    'pca_coords': coords_pca.tolist(),
                    'tsne_coords': getattr(self, 'tsne_snapshot', None), 
                    'preds': prd_snap.flatten().tolist(),
                    'y_true': self._y_raw[idx_snap].flatten().tolist()
                }
            except Exception as e_snap:
                if logger: logger(f"[AVISO] No se pudo generar snapshot: {e_snap}", "info")
                diag_snapshot = None

            metrics = {
                'confusion_matrix': cm.tolist(),
                'classification_report': rep,
                'diagnostic_snapshot': diag_snapshot,
                'auto_rating': auto_rating, # Nova Nota IA
                'generated_at': datetime.now().isoformat()
            }
            with open(os.path.join(tmpdir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            upload(os.path.join(tmpdir, 'metrics.json'), f"{CONFIG['bucket']}/{base_path}/metrics.json")
            
            # --- 2.6 Metadata já salva layers+num_input em metadata.json + weights.npz ---

            # --- 2.7 TensorBoard Projector Files (Auto-Snapshot) ---
            try:
                # Gerar arquivos do Projector usando a última amostra de treino (X_raw)
                self.save_projector_files(training_id, shortname, self._X_raw, self._y_raw, logger=logger)
            except Exception as e_proj:
                if logger: logger(f"[AVISO] Nota: No se pudo gerar archivos del Projector: {e_proj}", "info")

            # --- 2.8 Iteration History (Snapshots) ---
            if hasattr(self, 'snapshot_dir') and self.snapshot_dir and os.path.exists(self.snapshot_dir):
                if logger: logger("Sincronizando historial de iteraciones com GCS...", "info")
                # Sincroniza PNGs (legado/preview) e o novo snapshots_data.npz
                all_files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.png') or f == 'snapshots_data.npz']
                for sf in all_files:
                    src_sf = os.path.join(self.snapshot_dir, sf)
                    dest_sf = f"{CONFIG['bucket']}/{base_path}/history/{sf}"
                    upload(src_sf, dest_sf)

            # 3. Extracted Pixels
            if logger: logger("Guardar una matriz de píxeles en GCS...", "info")
            np.save(os.path.join(tmpdir, 'X_data.npy'), self._X_raw)
            np.save(os.path.join(tmpdir, 'y_data.npy'), self._y_raw)
            
            for fname in ['X_data.npy', 'y_data.npy']:
                src  = os.path.join(tmpdir, fname)
                dest = f"{CONFIG['bucket']}/{base_path}/extracted_pixels/{fname}"
                upload(src, dest)
                
        # 4. Copy samples
        if logger: logger("Copiar archivos CSV de las muestras para su almacenamiento...", "info")
        collections = getattr(self, '_sample_collections', [])
        for coll in collections:
            campaign = CONFIG.get('campaign', 'MONITOR_01')
            src = f"{CONFIG['bucket']}/{gcs_samples_path()}/{coll}.csv"
            dest = f"{CONFIG['bucket']}/{base_path}/samples/{coll}.csv"
            try:
                copy(src, dest)
            except Exception:
                pass

        self._last_saved_metadata = hp
        self._last_saved_metadata['metrics'] = metrics
        return hp

    @staticmethod
    def update_model_metadata(training_id, shortname, updates):
        """Atualiza campos específicos do metadata.json no GCS."""
        import json, tempfile
        from M_gcs import read_text, upload
        base_path = model_path(training_id, shortname)
        
        try:
            path = f"{base_path}/metadata.json"
            hp = json.loads(read_text(f"gs://{CONFIG['bucket']}/{path}"))
            hp.update(updates)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, 'metadata.json')
                with open(tmp_file, 'w') as f:
                    json.dump(hp, f, indent=2)
                upload(tmp_file, f"{CONFIG['bucket']}/{path}")
            return True
        except Exception as e:
            print(f"[ERR] Error updating metadata: {e}")
            return False

