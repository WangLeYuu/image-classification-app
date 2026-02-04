/**
 * 图像分类 API 模块
 * 
 * 提供与后端图像分类服务的通信功能
 */
import axios from 'axios'
import type { ClassificationResult, ApiError } from '@/types/classification'

// API 基础 URL
const API_BASE_URL = 'http://localhost:8001'

/**
 * 创建 axios 实例
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 秒超时
  headers: {
    'Content-Type': 'multipart/form-data'
  }
})

/**
 * 对图像进行分类
 * 
 * @param file - 要分类的图像文件
 * @returns 分类结果
 * @throws 当请求失败时抛出错误
 */
export const classifyImage = async (file: File): Promise<ClassificationResult> => {
  // 创建 FormData 对象
  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await apiClient.post<ClassificationResult>('/classify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      // 处理 Axios 错误
      const apiError = error.response?.data as ApiError | undefined
      const errorMessage = apiError?.detail || error.message
      throw new Error(`分类请求失败: ${errorMessage}`)
    }
    // 处理其他错误
    throw new Error('分类请求失败: 未知错误')
  }
}

/**
 * 健康检查
 * 
 * @returns 健康状态
 */
export const healthCheck = async (): Promise<{ status: string }> => {
  const response = await apiClient.get<{ status: string }>('/health')
  return response.data
}
