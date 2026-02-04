<template>
  <div class="image-classification">
    <el-card class="main-card">
      <template #header>
        <div class="card-header">
          <h1>
            <el-icon><Picture /></el-icon>
            图像分类应用
          </h1>
          <p class="subtitle">基于 MobileNetV2 的深度学习图像分类</p>
        </div>
      </template>

      <!-- 上传区域 -->
      <div class="upload-section">
        <el-upload
          class="image-uploader"
          drag
          action="#"
          :auto-upload="false"
          :show-file-list="false"
          :on-change="handleFileChange"
          accept=".jpg,.jpeg,.png,.bmp,.gif,.webp"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽图片到此处或 <em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持 JPG、PNG、BMP、GIF、WEBP 格式，文件大小不超过 10MB
            </div>
          </template>
        </el-upload>
      </div>

      <!-- 图片预览和结果区域 -->
      <div v-if="imageUrl" class="preview-section">
        <el-row :gutter="20">
          <!-- 图片预览 -->
          <el-col :xs="24" :sm="12">
            <div class="preview-container">
              <h3>
                <el-icon><View /></el-icon>
                图片预览
              </h3>
              <img :src="imageUrl" alt="预览图片" class="preview-image" />
            </div>
          </el-col>

          <!-- 分类结果 -->
          <el-col :xs="24" :sm="12">
            <div class="result-container">
              <h3>
                <el-icon><DataAnalysis /></el-icon>
                分类结果
              </h3>
              
              <!-- 加载状态 -->
              <div v-if="loading" class="loading-state">
                <el-skeleton :rows="3" animated />
                <p class="loading-text">
                  <el-icon class="is-loading"><Loading /></el-icon>
                  正在分析图像...
                </p>
              </div>

              <!-- 错误状态 -->
              <el-alert
                v-else-if="error"
                :title="error"
                type="error"
                :closable="false"
                show-icon
              />

              <!-- 结果展示 -->
              <div v-else-if="result" class="result-content">
                <!-- 主要预测结果 -->
                <div class="main-result">
                  <el-tag size="large" type="success" effect="dark" class="result-tag">
                    预测类别: {{ result.prediction.class_name }}
                  </el-tag>
                  <el-progress
                    :percentage="Math.round(result.prediction.confidence * 100)"
                    :color="progressColors"
                    :stroke-width="20"
                    class="confidence-progress"
                  >
                    <template #default="{ percentage }">
                      <span class="progress-label">置信度: {{ percentage }}%</span>
                    </template>
                  </el-progress>
                </div>

                <!-- Top-K 结果 -->
                <div class="top-k-results">
                  <h4>Top 5 预测结果</h4>
                  <el-table :data="result.top_k" style="width: 100%" size="small">
                    <el-table-column prop="class_name" label="类别" />
                    <el-table-column prop="confidence" label="置信度" width="120">
                      <template #default="scope">
                        {{ (scope.row.confidence * 100).toFixed(2) }}%
                      </template>
                    </el-table-column>
                    <el-table-column label="可视化" width="150">
                      <template #default="scope">
                        <el-progress
                          :percentage="Math.round(scope.row.confidence * 100)"
                          :color="progressColors"
                          :show-text="false"
                          :stroke-width="10"
                        />
                      </template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>

              <!-- 初始状态 -->
              <div v-else class="empty-state">
                <el-icon :size="48" color="#909399"><Picture /></el-icon>
                <p>上传图片后将自动进行分类</p>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
/**
 * 图像分类视图组件
 * 
 * 提供图像上传、预览和分类结果展示功能
 * 与后端 API 通信进行图像分类
 */
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import type { UploadFile } from 'element-plus'
import { classifyImage } from '@/api/classification'
import type { ClassificationResult } from '@/types/classification'

// 响应式数据
const imageUrl = ref<string>('')
const loading = ref<boolean>(false)
const error = ref<string>('')
const result = ref<ClassificationResult | null>(null)

// 进度条颜色配置
const progressColors = [
  { color: '#f56c6c', percentage: 20 },
  { color: '#e6a23c', percentage: 40 },
  { color: '#5cb87a', percentage: 60 },
  { color: '#1989fa', percentage: 80 },
  { color: '#6f7ad3', percentage: 100 }
]

/**
 * 处理文件选择变化
 * 
 * @param uploadFile - 上传的文件对象
 */
const handleFileChange = async (uploadFile: UploadFile) => {
  const rawFile = uploadFile.raw
  
  if (!rawFile) {
    ElMessage.error('文件读取失败')
    return
  }

  // 验证文件类型
  const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif', 'image/webp']
  if (!allowedTypes.includes(rawFile.type)) {
    ElMessage.error('不支持的文件格式，请上传图片文件')
    return
  }

  // 验证文件大小 (10MB)
  const maxSize = 10 * 1024 * 1024
  if (rawFile.size > maxSize) {
    ElMessage.error('文件大小超过 10MB 限制')
    return
  }

  // 创建图片预览 URL
  imageUrl.value = URL.createObjectURL(rawFile)
  
  // 重置状态
  error.value = ''
  result.value = null
  
  // 执行分类
  await performClassification(rawFile)
}

/**
 * 执行图像分类
 * 
 * @param file - 要分类的图像文件
 */
const performClassification = async (file: File) => {
  loading.value = true
  error.value = ''
  
  try {
    const response = await classifyImage(file)
    result.value = response
    ElMessage.success('图像分类完成')
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : '分类失败，请重试'
    error.value = errorMessage
    ElMessage.error(errorMessage)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.image-classification {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.main-card {
  min-height: 600px;
}

.card-header {
  text-align: center;
}

.card-header h1 {
  margin: 0;
  font-size: 28px;
  color: #303133;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.subtitle {
  margin: 10px 0 0;
  color: #909399;
  font-size: 14px;
}

.upload-section {
  margin: 30px 0;
}

.image-uploader {
  text-align: center;
}

.image-uploader .el-upload {
  width: 100%;
}

.image-uploader .el-upload-dragger {
  width: 100%;
  height: 250px;
}

.el-icon--upload {
  font-size: 48px;
  color: #409eff;
  margin-bottom: 10px;
}

.el-upload__text {
  font-size: 16px;
  color: #606266;
}

.el-upload__text em {
  color: #409eff;
  font-style: normal;
}

.el-upload__tip {
  margin-top: 10px;
  color: #909399;
}

.preview-section {
  margin-top: 30px;
}

.preview-container,
.result-container {
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
  min-height: 400px;
}

.preview-container h3,
.result-container h3 {
  margin: 0 0 20px;
  font-size: 18px;
  color: #303133;
  display: flex;
  align-items: center;
  gap: 8px;
}

.preview-image {
  width: 100%;
  max-height: 350px;
  object-fit: contain;
  border-radius: 8px;
  background-color: #fff;
}

.loading-state {
  text-align: center;
  padding: 40px 20px;
}

.loading-text {
  margin-top: 20px;
  color: #409eff;
  font-size: 14px;
}

.is-loading {
  animation: rotating 2s linear infinite;
  margin-right: 8px;
}

@keyframes rotating {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #909399;
}

.empty-state p {
  margin-top: 16px;
  font-size: 14px;
}

.main-result {
  text-align: center;
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  margin-bottom: 20px;
}

.result-tag {
  font-size: 16px;
  padding: 10px 20px;
  margin-bottom: 15px;
}

.confidence-progress {
  margin-top: 15px;
}

.progress-label {
  font-size: 14px;
  color: #606266;
}

.top-k-results {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
}

.top-k-results h4 {
  margin: 0 0 15px;
  font-size: 16px;
  color: #303133;
}

@media (max-width: 768px) {
  .image-classification {
    padding: 10px;
  }
  
  .card-header h1 {
    font-size: 22px;
  }
  
  .preview-container,
  .result-container {
    margin-bottom: 20px;
    min-height: auto;
  }
}
</style>
