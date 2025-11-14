# Redis Setup Guide

## 🚀 Start Redis

### Step 1: Open PowerShell

### Step 2: Run
```powershell
cd D:\client-project\redis
.\redis-server.exe redis.windows.conf
```

### Step 3: Test (in new PowerShell window)
```powershell
cd D:\client-project\redis
.\redis-cli.exe ping
```
**Expected output:** `PONG`

---

## 🛑 Stop Redis

### Option 1: Close the Redis window

### Option 2: Run this command
```powershell
Stop-Process -Name "redis-server" -Force
```

---

## ✅ Check if Redis is Running

```powershell
cd D:\client-project\redis
.\redis-cli.exe ping
```

If running: `PONG` ✅  
If not running: `Connection refused` ❌

---

## 📝 Notes

- **Keep Redis window open** while using the chatbot
- Redis location: `D:\client-project\redis`
- Default port: `6379`

