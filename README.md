# Latent Merging Playground

Qwen2.5-7B-Instruct(베이스)와 OpenThinker3-7B(FT) 사이에서 네 가지 병합 기법을 실험하기 위한 최소 코드/요약 결과 정리본입니다. 대용량 산출물(pkl, safetensors, 체크포인트)은 포함하지 않고, 경로/배치 방법만 안내합니다.

## 구성
- `src/latent_merging.py` — LERP / SLERP / RegMean / Task Vector(Δ 주입) 핵심 클래스와 생성 유틸.
- `src/metrics.py` — CKA, midness(구면 중점 기준) 등 평가 지표.
- `scripts/judgebench_eval.py` — JudgeBench용 A/B 응답 평가 스크립트(LLM 심판; OpenAI API 키는 환경변수로 주입).
- `results/` — 실험에서 나온 요약 CSV(기존 결과).
- `artifacts/` — 실제 실험 산출물 pkl 포함(영문 파일명으로 정리):
  - `artifacts/SLERP/`, `artifacts/Lerp/`, `artifacts/RegMean/`에 scale×step별 raw(`*.pkl`)과 요약(`*summary.pkl`) pkl
  - `artifacts/TaskVector/`에 Task Vector 합성(`latent_merged-TaskVector.pkl`, `weight_merged-TaskVector.pkl`)
  - `artifacts/root_pkls/`에 기타 응답/평가 pkl(`Qwen_baseline.pkl`, `OpenThinker_baseline.pkl`, `claude_latent_merging.pkl` 등)과 합성 가중치(`latent_merged-RegMean.pkl`, `weight_merged-RegMean.pkl`, `weight_merged-lerp.pkl`).

## 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
필수: `torch`, `transformers`, `datasets`, `tqdm`, `pandas`, `openai`(v1) 등.

## 대용량 파일 배치(별도 다운로드/복사)
- 모델 체크포인트: `헬스케어_모델/` 등은 루트나 원하는 경로에 두고, 스크립트 인자로 경로 지정.
- 응답/병합 결과: `SLERP/`, `Lerp/`, `RegMean/`, `Task Vector/`에 있는 `*.pkl`을 필요 시 복사해 동일 경로에 두면 평가 스크립트가 읽을 수 있습니다.
- 합성 가중치: `weight_merged-*.pkl`, `latent_merged-*.pkl` 파일도 동일하게 별도 보관 후 참조 경로를 넘기세요.

## 사용 예시
- JudgeBench 평가(응답 pkl 두 개 비교):
  ```bash
  OPENAI_API_KEY=... python scripts/judgebench_eval.py \
    --path-a SLERP/0.5_0.pkl \
    --path-b Openthinker_비교용.pkl \
    --judge-model gpt-4o-mini \
    --out results/judgebench_eval.jsonl
  ```
- LERP/SLERP 생성:
  ```python
  from src.latent_merging import get_model, get_tokenizer, latent_mix_generate
  tok = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")
  base = get_model("Qwen/Qwen2.5-7B-Instruct")
  ft = get_model("open-thoughts/OpenThinker3-7B")
  msgs = [{"role":"user","content":"간단히 자기소개해줘."}]
  text = latent_mix_generate(base, ft, tok, messages=msgs, mix_layer=20, beta=0.5, mode="slerp")
  print(text)
  ```
- RegMean/Task Vector는 동일 파일에서 `latent_regmean_generate`, `delta_generate`/`ActivationSteerer`를 참고하세요.

## 주의
- API 키를 하드코딩하지 마세요(환경변수 `OPENAI_API_KEY` 사용).
- 결과 재현을 위해선 JudgeBench 데이터셋(`datasets` 라이브러리)과 위 경로의 pkl 산출물이 필요합니다. 큰 파일은 GitHub에 올리지 말고 외부 스토리지를 활용하세요.***
