import { createRouter, createWebHistory } from 'vue-router'
import ImageClassificationView from '@/views/ImageClassificationView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: ImageClassificationView
    }
  ]
})

export default router
