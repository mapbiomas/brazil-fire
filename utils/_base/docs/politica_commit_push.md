# Política de commit e push

**Regra:** nenhum commit nem push é feito sem autorização explícita de quem mantém o repositório para aquela alteração. Vale para pessoas e para agentes de código (Claude Code etc.).

- Autorização para uma alteração não vale para a seguinte.
- Alterações são preparadas e mostradas (lista de arquivos e resumo) antes de pedir autorização.
- Commits pequenos, fiéis aos arquivos originais, com mensagem que diz o que mudou e por quê.
- Nunca incluir credenciais, tokens ou IDs de assets privados.
- Antes do commit: rodar os testes da pasta alterada (ver `AGENTS.md`).

Esta regra estava só em `utils/fire_monitor_highlights/CLAUDE.md`; passa a valer para o repositório inteiro.

**Nota:** autorizações permanentes dadas pela mantenedora fora do repositório (por exemplo, "pode ir direto para a `main`") continuam valendo onde foram dadas, mas devem ser registradas aqui se quiserem virar regra do repositório. Decisão em aberto.
