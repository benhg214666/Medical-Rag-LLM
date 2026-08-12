# Medical RAG Frontend

React、Vite 與 TypeScript 製作的本機醫療 RAG 操作介面。後端必須另外啟動。

## 開發環境

- Node.js 18 或更新版本
- 已安裝並可啟動的專案 FastAPI 後端

```bash
# 在專案根目錄啟動後端
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 在另一個終端機啟動前端
cd frontend
npm install
cp .env.example .env
npm run dev
```

瀏覽 `http://localhost:5173`。若後端位址不同，請在 `.env` 設定：

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

正式建置：

```bash
cd frontend
npm run build
```

文件列表只保留在目前頁面的記憶體中，重新整理後會清除；後端目前沒有文件列表 API。
