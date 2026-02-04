/**
 * 图像分类相关的类型定义
 */

/**
 * 单个预测结果
 */
export interface PredictionResult {
  /** 类别名称 */
  class_name: string
  /** 置信度分数 (0-1) */
  confidence: number
}

/**
 * 图像分类 API 响应结果
 */
export interface ClassificationResult {
  /** 是否成功 */
  success: boolean
  /** 文件名 */
  filename: string
  /** 主要预测结果 */
  prediction: PredictionResult
  /** Top-K 预测结果列表 */
  top_k: PredictionResult[]
}

/**
 * API 错误响应
 */
export interface ApiError {
  /** 错误详情 */
  detail: string
}
