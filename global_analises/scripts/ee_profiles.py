"""Perfis de login do Google Earth Engine (várias contas na mesma máquina).

O earthengine guarda um único login em ~/.config/earthengine/credentials.
Este módulo guarda cópias nomeadas em ~/.config/earthengine/profiles/<nome>.json
e inicializa o ee com a conta escolhida, sem sobrescrever o login padrão.

Linha de comando:
    python ee_profiles.py list                    # perfis salvos
    python ee_profiles.py save workspace2         # salva o login padrão atual como perfil
    python ee_profiles.py login workspace3        # abre o navegador, loga e salva como perfil
                                                  # (o login padrão anterior é restaurado)
    python ee_profiles.py test workspace3 --project <projeto-cloud>

Em código:
    from ee_profiles import init
    init('workspace3', project='meu-projeto')
"""
import argparse
import json
import pathlib
import shutil

import ee

EE_DIR = pathlib.Path.home() / '.config' / 'earthengine'
DEFAULT = EE_DIR / 'credentials'
PROFILES = EE_DIR / 'profiles'


def path(name):
    return PROFILES / f'{name}.json'


def save(name):
    PROFILES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEFAULT, path(name))
    print('perfil salvo:', path(name))


def login(name):
    backup = EE_DIR / 'credentials.bak_profiles'
    had = DEFAULT.exists()
    if had:
        shutil.copy2(DEFAULT, backup)
    try:
        ee.Authenticate(force=True, auth_mode='localhost')
        save(name)
    finally:
        if had:
            shutil.copy2(backup, DEFAULT)
            backup.unlink()
            print('login padrão anterior restaurado')


def credentials(name):
    from google.oauth2.credentials import Credentials
    d = json.loads(path(name).read_text())
    return Credentials(None, refresh_token=d['refresh_token'], client_id=d['client_id'],
                       client_secret=d['client_secret'], token_uri='https://oauth2.googleapis.com/token',
                       scopes=d.get('scopes'))


def init(name=None, project=None):
    """Inicializa o ee com o perfil `name` (None = login padrão)."""
    if name:
        ee.Initialize(credentials(name), project=project)
    else:
        ee.Initialize(project=project)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'save', 'login', 'test'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--project')
    a = ap.parse_args()
    if a.cmd == 'list':
        for p in sorted(PROFILES.glob('*.json')):
            print(p.stem)
    elif a.cmd == 'save':
        save(a.name)
    elif a.cmd == 'login':
        login(a.name)
    elif a.cmd == 'test':
        init(a.name, a.project)
        n = ee.Number(1).add(1).getInfo()
        ops = ee.data.listOperations()
        running = sum(o['metadata']['state'] in ('PENDING', 'RUNNING') for o in ops)
        print(f'ok ({n}) · perfil {a.name} · projeto {a.project} · tasks na fila/rodando: {running}')


if __name__ == '__main__':
    main()
