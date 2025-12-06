# BigQuery Setup Guide

## Why BigQuery?

BigQuery를 사용하면 GDELT 데이터를 **10-100배 빠르게** 수집할 수 있습니다:

- **Direct Download**: 10년치 데이터 수집에 수 시간 소요
- **BigQuery**: 10년치 데이터 수집에 수 분 소요

BigQuery는 Google Cloud의 관리형 데이터 웨어하우스로, GDELT 전체 데이터셋을 호스팅하고 있습니다.

## Prerequisites

1. Google Cloud Platform (GCP) 계정
2. 결제 정보 등록 (무료 크레딧 $300 사용 가능)
3. BigQuery API 활성화된 프로젝트

## Setup Options

### Option 1: Application Default Credentials (권장)

가장 간단한 방법입니다.

#### Step 1: gcloud CLI 설치

**Windows:**
```bash
# Download from: https://cloud.google.com/sdk/docs/install
```

**Mac/Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

#### Step 2: 로그인 및 인증

```bash
# GCP 로그인
gcloud auth login

# Application default credentials 설정
gcloud auth application-default login

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID
```

#### Step 3: 패키지 설치

```bash
pip install google-cloud-bigquery google-auth db-dtypes
```

#### Step 4: 실행

```bash
# config.yaml에서 method를 bigquery로 설정하거나
python main.py --bigquery

# 또는 특정 지역만 테스트
python src/data_collection/fetch_gdelt_bigquery.py --limit 1000 --regions israel_palestine
```

### Option 2: Service Account (프로덕션 환경)

보다 안전한 방법으로, CI/CD나 서버 환경에 적합합니다.

#### Step 1: Service Account 생성

1. GCP Console → IAM & Admin → Service Accounts
2. "Create Service Account" 클릭
3. 이름 입력 (예: `gdelt-fetcher`)
4. Role: `BigQuery User` 권한 부여
5. "Create Key" → JSON 선택 → 다운로드

#### Step 2: Credentials 설정

다운로드한 JSON 파일을 프로젝트에 저장:

```bash
mkdir -p credentials
mv ~/Downloads/your-service-account.json credentials/gcp-credentials.json
```

**중요**: `.gitignore`에 추가:
```
credentials/
*.json
```

#### Step 3: config.yaml 수정

```yaml
data_collection:
  method: "bigquery"
  bigquery:
    credentials_path: "credentials/gcp-credentials.json"
    project_id: "your-project-id"
    batch_months: 3
```

#### Step 4: 실행

```bash
python main.py --bigquery
```

## Usage Examples

### 테스트 쿼리 (소량 데이터)

```bash
# 1000개 행만 가져오기 (테스트용)
python src/data_collection/fetch_gdelt_bigquery.py --limit 1000 --regions israel_palestine
```

### 전체 데이터 수집

```bash
# 모든 지역
python main.py --bigquery

# 특정 지역만
python main.py --bigquery --regions israel_palestine russia_ukraine
```

### Direct Download로 돌아가기

```bash
# 명령줄 옵션
python main.py --no-bigquery

# 또는 config.yaml 수정
data_collection:
  method: "download"
```

## Query Customization

`src/data_collection/fetch_gdelt_bigquery.py`의 `build_query` 메서드를 수정하여 쿼리를 커스터마이즈할 수 있습니다:

```python
def build_query(self, start_date, end_date, countries, actor_keywords, limit=None):
    query = f"""
    SELECT
        *  -- 또는 필요한 컬럼만 선택
    FROM
        `gdelt-bq.gdeltv2.events`
    WHERE
        SQLDATE BETWEEN {start_date_int} AND {end_date_int}
        AND (country_filter OR keyword_filter)
        AND GoldsteinScale < -5  -- 예: 추가 필터링
    """
    return query
```

## Cost Estimation

BigQuery는 쿼리당 비용을 청구합니다:

- **가격**: $5 per TB scanned
- **무료 할당량**: 1 TB/month

**예상 비용**:
- 10년 데이터 쿼리: ~5-10 GB 스캔
- 비용: ~$0.03-0.05 (거의 무료)
- 무료 크레딧으로 충분히 커버 가능

## Troubleshooting

### Error: "google.cloud.bigquery not found"

```bash
pip install google-cloud-bigquery google-auth db-dtypes
```

### Error: "Could not automatically determine credentials"

**Option 1**: Application default credentials 설정
```bash
gcloud auth application-default login
```

**Option 2**: Service account 사용
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### Error: "Permission denied"

Service account에 `BigQuery User` 권한이 있는지 확인:
1. GCP Console → IAM & Admin → IAM
2. Service account 찾기
3. "BigQuery User" role 추가

### Timeout Errors

큰 날짜 범위를 쿼리하면 timeout이 발생할 수 있습니다. `config.yaml`에서 배치 크기를 줄이세요:

```yaml
data_collection:
  bigquery:
    batch_months: 1  # 3에서 1로 줄임
```

## Performance Tips

1. **날짜 범위 최적화**: 필요한 기간만 쿼리
2. **컬럼 선택**: `SELECT *` 대신 필요한 컬럼만 선택
3. **파티션 활용**: SQLDATE로 자동 파티셔닝됨
4. **캐싱**: 같은 쿼리는 캐시에서 가져옴 (무료)

## Comparison: BigQuery vs Direct Download

| Feature | BigQuery | Direct Download |
|---------|----------|----------------|
| Speed | ⚡⚡⚡ 매우 빠름 (분) | 🐌 느림 (시간) |
| Setup | 🔧 GCP 계정 필요 | ✅ 설정 불필요 |
| Cost | 💰 거의 무료 ($0.03) | 🆓 완전 무료 |
| Filtering | ✅ SQL로 서버측 필터링 | ❌ 다운로드 후 필터링 |
| Recommended | ✅ 프로덕션/연구용 | ✅ 테스트/소규모 |

## Next Steps

1. BigQuery 설정 완료
2. 테스트 쿼리 실행: `--limit 1000`
3. 전체 데이터 수집: `python main.py --bigquery`
4. 모델 학습 시작!

## Resources

- [Google Cloud BigQuery 문서](https://cloud.google.com/bigquery/docs)
- [GDELT BigQuery 테이블](https://console.cloud.google.com/marketplace/product/gdelt-bq/gdelt-2)
- [BigQuery Python Client](https://googleapis.dev/python/bigquery/latest/index.html)
