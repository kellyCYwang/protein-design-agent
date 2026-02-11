# Kubernetes Deployment for Protein RAG Agent

This directory contains Kubernetes manifests for deploying the Protein Design Agent to a Kubernetes cluster.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ingress (HTTPS)                          │
│                    protein-rag.your-domain.com                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   protein-rag-app (2 replicas)                  │
│                     Streamlit UI + Agent                        │
│                         Port 8501                               │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ embedding-service │ │   PVC: papers   │ │   PVC: chroma/bm25  │
│   PubMedBERT      │ │   (shared PDF)  │ │   (vector index)    │
│    Port 8000      │ └─────────────────┘ └─────────────────────┘
└───────────────────┘
```

## Prerequisites

1. **Kubernetes cluster** (1.24+)
2. **kubectl** configured for your cluster
3. **Container registry** (Docker Hub, ECR, GCR, etc.)
4. **Ingress controller** (nginx-ingress recommended)
5. **cert-manager** (for TLS certificates, optional)
6. **Storage class** supporting ReadWriteMany (NFS, EFS, etc.)

## Quick Start

### 1. Build and push images

```bash
# Build main app
docker build -t your-registry/protein-rag-app:latest -f Dockerfile .
docker push your-registry/protein-rag-app:latest

# Build embedding service
docker build -t your-registry/protein-rag-embedding:latest -f Dockerfile.embedding .
docker push your-registry/protein-rag-embedding:latest
```

### 2. Configure secrets

Edit `k8s/secret.yaml` with your actual API keys:

```yaml
stringData:
  PINECONE_API_KEY: "your-actual-pinecone-key"
  ZENMUX_API_KEY: "your-actual-zenmux-key"
```

**⚠️ Important:** Never commit secrets to git! Use:
- [sealed-secrets](https://github.com/bitnami-labs/sealed-secrets)
- [external-secrets](https://external-secrets.io/)
- Cloud provider secret managers (AWS Secrets Manager, etc.)

### 3. Update image references

Edit `k8s/kustomization.yaml` to point to your registry:

```yaml
images:
  - name: your-registry/protein-rag-app
    newName: my-registry.example.com/protein-rag-app
    newTag: v1.0.0
```

### 4. Update Ingress hostname

Edit `k8s/ingress.yaml` and replace `protein-rag.your-domain.com` with your actual domain.

### 5. Deploy

```bash
# Apply all manifests
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment-embedding.yaml
kubectl apply -f k8s/deployment-app.yaml
kubectl apply -f k8s/ingress.yaml
```

### 6. Upload PDFs

```bash
# Copy PDFs to the papers PVC (adjust pod name)
kubectl cp ./data/papers/ protein-rag/protein-rag-app-xxx:/app/data/papers/
```

### 7. Run initial indexing

```bash
# Create indexing job
kubectl apply -f k8s/job-indexing.yaml

# Monitor progress
kubectl logs -f job/protein-rag-index -n protein-rag
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_BACKEND` | Vector DB backend ("chroma" or "pinecone") | `chroma` |
| `RAG_PDF_DIR` | PDF storage directory | `/app/data/papers` |
| `RAG_CHROMA_DIR` | ChromaDB persist directory | `/app/data/chroma_db` |
| `RAG_BM25_DIR` | BM25 index directory | `/app/data/bm25` |
| `RAG_IMAGE_DIR` | Extracted images directory | `/app/data/pdf_images` |
| `EMBEDDING_SERVICE_URL` | Embedding service URL | `http://embedding-service:8000/embed` |
| `PINECONE_API_KEY` | Pinecone API key | (required if using Pinecone) |
| `PINECONE_INDEX_NAME` | Pinecone index name | `paper-rag-index` |
| `ZENMUX_API_KEY` | LLM API key | (required) |

### Scaling

- **App replicas**: Edit `deployment-app.yaml` → `spec.replicas` (2-3 for ~10 users)
- **Embedding replicas**: Edit `deployment-embedding.yaml` → `spec.replicas` (1 is usually enough)

### Storage

The deployment uses PersistentVolumeClaims with `ReadWriteMany` access mode. Ensure your cluster has a storage class that supports this (e.g., NFS, EFS, Azure Files).

If your cluster only supports `ReadWriteOnce`:
1. Set app replicas to 1, OR
2. Use Pinecone (cloud vector DB) instead of ChromaDB

## Monitoring

```bash
# Check pod status
kubectl get pods -n protein-rag

# View app logs
kubectl logs -f deployment/protein-rag-app -n protein-rag

# View embedding service logs
kubectl logs -f deployment/embedding-service -n protein-rag

# Check ingress
kubectl describe ingress protein-rag-ingress -n protein-rag
```

## Troubleshooting

### Pods stuck in Pending
- Check PVC binding: `kubectl get pvc -n protein-rag`
- Ensure storage class supports ReadWriteMany

### Embedding service not ready
- Model download can take 2-5 minutes on first start
- Check logs: `kubectl logs -f deployment/embedding-service -n protein-rag`

### 502 Bad Gateway
- Check if pods are running: `kubectl get pods -n protein-rag`
- Check readiness probes: `kubectl describe pod <pod-name> -n protein-rag`

### WebSocket errors (Streamlit)
- Ensure Ingress has WebSocket support annotations (see `ingress.yaml`)
