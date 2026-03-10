# Guia de Configuração de Credenciais (Windows)

Este guia explica como configurar o acesso ao **GitHub** e ao **Google Source (Earth Engine)** do zero.

---

## 1. Configuração de Identidade Básica
Abra o PowerShell e execute estes comandos com seus dados:

```powershell
git config --global user.name "wallyboy22"
git config --global user.email "wallace.silva@ipam.org.br"
```

---

## 2. Acesso ao GitHub (Personal Access Token)
Como a autenticação por senha comum foi desativada, você precisa de um Token (PAT).

1. Vá em: [GitHub Settings -> Tokens (classic)](https://github.com/settings/tokens)
2. Clique em **Generate new token (classic)**.
3. Marque a caixinha **'repo'** (completo).
4. Clique em **Generate token** e **COPIE** o código (ele começa com `ghp_`).
5. No terminal, execute:
   ```powershell
   git config --global credential.helper manager
   ```
6. Agora tente puxar algo:
   ```powershell
   git fetch origin
   ```
   *Uma janela do Windows aparecerá. Use seu usuário `wallyboy22` e o **Token** que você copiou como senha.*

---

## 3. Acesso ao Google Source (Earth Engine Scripts)
Os scripts do IPAM no GEE usam um sistema de cookies.

1. Acesse: [https://earthengine.googlesource.com/new-password](https://earthengine.googlesource.com/new-password)
2. Faça login com sua conta IPAM/institucional.
3. Você verá um bloco de código no navegador. **Copie apenas a linha que começa com `earthengine.googlesource.com...`**.
4. Crie o arquivo de cookies localmente na pasta do projeto:
   ```powershell
   # Cole sua linha entre as aspas abaixo
   $content = "COLE_AQUI_A_LINHA_DO_NAVEGADOR"
   Set-Content -Path .gitcookies -Value $content
   ```
5. Configure o Git para olhar este arquivo nesta pasta:
   ```powershell
   git config http.cookiefile .gitcookies
   ```

---

## 4. Teste de Funcionamento
Execute estes comandos para confirmar que você tem acesso a tudo:

### Testar GitHub:
```powershell
git remote show origin
```
*(Deve retornar "up to date" ou listar branches sem erro)*

### Testar Google Source:
```powershell
git ls-remote https://earthengine.googlesource.com/users/geomapeamentoipam/mapeamento_fogo
```
*(Deve listar uma série de códigos/hashes e refs/heads/master)*

---

**Nota Importante:** O arquivo `.gitcookies` é secreto. Já o adicionei ao seu `.gitignore` para que ele nunca seja enviado para o servidor.
