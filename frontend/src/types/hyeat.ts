/**
 * HY-eat 대기 시스템 DynamoDB 아이템 타입 정의
 *
 * @description
 * Jetson Nano에서 측정한 식당 대기 데이터를 DynamoDB에 저장하는 형식.
 * snake_case(Python) → camelCase(DynamoDB/TypeScript) 변환 완료된 형태.
 *
 * @example
 * ```typescript
 * const item: DdbWaitingItem = {
 *   pk: "CORNER#hanyang_plaza#korean",
 *   sk: "1770349800000",
 *   restaurantId: "hanyang_plaza",
 *   cornerId: "korean",
 *   queueLen: 15,
 *   estWaitTimeMin: 8,
 *   dataType: "observed",
 *   source: "jetson_nano",
 *   timestampIso: "2026-02-15T17:00:00+09:00",
 *   createdAtIso: "2026-02-15T17:00:01+09:00",
 *   ttl: 1772941800,
 * };
 * ```
 */
interface DdbWaitingItem {
  // [Core Identifiers]

  /** Partition Key: "CORNER#{restaurantId}#{cornerId}" */
  pk: string;

  /** Sort Key: epoch milliseconds (문자열) */
  sk: string;

  // [Data Fields]

  /** 식당 ID (예: "hanyang_plaza") */
  restaurantId: string;

  /** 코너 ID (예: "korean", "japanese", "chinese") */
  cornerId: string;

  /** 현재 대기 인원 수 */
  queueLen: number;

  /** 예상 대기 시간 (분 단위) */
  estWaitTimeMin: number;

  // [Meta Data]

  /** 데이터 타입: 실측값 | 예측값 | 더미 */
  dataType: "observed" | "predicted" | "dummy";

  /** 데이터 소스 (예: "jetson_nano") */
  source?: string;

  // [Human Readable Dates]

  /** 젯슨 측정 시간 ISO 8601 (KST, 예: "2026-02-13T12:00:00+09:00") */
  timestampIso: string;

  /** DB 저장 시간 ISO 8601 (KST, 예: "2026-02-13T12:00:01+09:00") */
  createdAtIso: string;

  // [System Management]

  /** TTL: DynamoDB 자동 삭제용 epoch timestamp (초 단위) */
  ttl?: number;
}

/**
 * API 응답 타입 (프론트엔드에서 사용)
 *
 * @description
 * DynamoDB에서 조회한 데이터를 프론트엔드에 전달할 때 사용하는 간소화된 형식.
 * PK/SK 등 DynamoDB 내부 키는 제외.
 *
 * @example
 * ```typescript
 * const response: WaitingDataResponse = {
 *   cornerId: "korean",
 *   queueLen: 15,
 *   estWaitTimeMin: 8,
 *   lastUpdated: "2026-02-15T17:00:00+09:00",
 * };
 * ```
 */
interface WaitingDataResponse {
  /** 코너 ID (예: "korean") */
  cornerId: string;

  /** 현재 대기 인원 수 */
  queueLen: number;

  /** 예상 대기 시간 (분 단위) */
  estWaitTimeMin: number;

  /** 마지막 업데이트 시간 ISO 8601 */
  lastUpdated: string;
}

/**
 * 코너별 실시간 대기 현황
 *
 * @description
 * 프론트엔드 대시보드에서 각 코너의 현재 대기 상태를 표시할 때 사용.
 * status 필드로 색상/아이콘 등 시각적 표현을 결정.
 *
 * @example
 * ```typescript
 * const corner: CornerStatus = {
 *   cornerId: "korean",
 *   cornerName: "한식 코너",
 *   queueLen: 15,
 *   estWaitTimeMin: 8,
 *   status: "crowded",
 *   lastUpdated: "2026-02-15T17:00:00+09:00",
 * };
 *
 * // 상태별 색상 매핑 예시
 * const statusColor: Record<CornerStatus["status"], string> = {
 *   available: "#4CAF50",  // 초록
 *   crowded: "#FF9800",    // 주황
 *   full: "#F44336",       // 빨강
 * };
 * ```
 */
interface CornerStatus {
  /** 코너 ID (예: "korean") */
  cornerId: string;

  /** 코너 한글 이름 (예: "한식 코너") */
  cornerName: string;

  /** 현재 대기 인원 수 */
  queueLen: number;

  /** 예상 대기 시간 (분 단위) */
  estWaitTimeMin: number;

  /** 대기 상태: 여유 | 혼잡 | 만석 */
  status: "available" | "crowded" | "full";

  /** 마지막 업데이트 시간 ISO 8601 */
  lastUpdated: string;
}

export type { DdbWaitingItem, WaitingDataResponse, CornerStatus };
