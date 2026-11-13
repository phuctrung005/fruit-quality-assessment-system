import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import tensorflow as tf
import numpy as np
import cv2
import os

# ----------------- CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH DỘNG -----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PATH_MODEL_NHAN_DIEN = os.path.join(PROJECT_ROOT, "Models", "Model_NhanDIen", "faster_rcnn_best_model.pth")
PATH_MODEL_PHAN_LOAI = os.path.join(PROJECT_ROOT, "Models", "Model_PhanLoai", "fruit_ripeness_model.h5")

# ----------------- THÔNG TIN LỚP (CLASSES) -----------------
CLASS_NAMES_NHAN_DIEN = {
    0: 'FruitDisease2', 
    1: 'Alternaria xoài (alternaria_mango)', 
    2: 'Thán thư ổi (anthracnose_guava)', 
    3: 'Thán thư xoài (anthracnose_mango)', 
    4: 'Thối đen xoài (black_mould_rot_mango)', 
    5: 'Đốm đen táo (blotch_apple)', 
    6: 'Ruồi đục quả ổi (fruitfly_guava)', 
    7: 'Ổi khỏe mạnh (healthy_guava)', 
    8: 'Xoài khỏe mạnh (healthy_mango)', 
    9: 'Táo khỏe mạnh (healthy_apple)', 
    10: 'Thối táo (rot_apple)', 
    11: 'Ghẻ táo (scab_apple)',
    12: 'Background'
}

LABELS_PHAN_LOAI = ['Chín nẫu (Overripe)', 'Chín (Ripe)', 'Chưa chín (Unripe)']

# ----------------- THIẾT LẬP TRANG STREAMLIT -----------------
st.set_page_config(
    page_title="Hệ Thống Đánh Giá Chất Lượng Hoa Quả",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS để tăng tính thẩm mỹ giao diện (Premium Style)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #333333;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 5px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF8E53;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B, #FF8E53) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        padding: 10px 24px !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍎 HỆ THỐNG ĐÁNH GIÁ CHẤT LƯỢNG HOA QUẢ</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Tích hợp phân loại độ chín (TensorFlow) & nhận diện lỗi/bệnh hư hỏng (PyTorch)</div>", unsafe_allow_html=True)

# ----------------- TẢI MÔ HÌNH (CACHED) -----------------
@st.cache_resource
def load_classification_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_detection_model(path, num_classes):
    # Khởi tạo mô hình Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model, device

# Nạp các model
with st.spinner("Đang tải các mô hình AI... Vui lòng đợi trong giây lát."):
    try:
        model_cls = load_classification_model(PATH_MODEL_PHAN_LOAI)
        model_det, det_device = load_detection_model(PATH_MODEL_NHAN_DIEN, num_classes=13)
        st.sidebar.success("✅ Đã tải các mô hình thành công!")
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
        st.info("Vui lòng kiểm tra lại sự tồn tại của các file trong thư mục 'Models/'.")

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.header("⚙️ Cấu hình bộ lọc")
threshold = st.sidebar.slider(
    "Ngưỡng tin cậy nhận diện lỗi (Confidence Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# ----------------- XỬ LÝ HÌNH ẢNH & THIẾT LẬP SESSION STATE -----------------
uploaded_file = st.file_uploader("📸 Tải ảnh hoa quả lên để phân tích...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Đọc ảnh
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    # Reset session state khi đổi file ảnh mới
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.classification_done = False
        st.session_state.ripeness_result = None
        st.session_state.ripeness_probs = None
        st.session_state.run_detection = False
        st.session_state.detection_done = False
        st.session_state.detection_preds = None

    # Hiển thị layout chính
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.markdown("<div class='step-header'>📷 Ảnh đầu vào</div>", unsafe_allow_html=True)
        st.image(img, caption="Ảnh gốc được tải lên", use_column_width=True)

    with col_results:
        # ================= STEP 1: PHÂN LOẠI ĐỘ CHÍN (TỰ ĐỘNG) =================
        st.markdown("<div class='step-header'>1️⃣ Phân loại Độ chín</div>", unsafe_allow_html=True)
        
        if not st.session_state.classification_done:
            with st.spinner("Đang chạy mô hình phân loại độ chín..."):
                # Tiền xử lý ảnh cho mô hình Keras (Resize về 224x224 và chuẩn hóa [0, 1])
                img_resized = cv2.resize(img_np, (224, 224))
                img_array = img_resized.astype('float32') / 255.0
                img_final = np.expand_dims(img_array, axis=0)
                
                # Dự đoán
                prediction = model_cls.predict(img_final, verbose=0)[0]
                pred_class_idx = np.argmax(prediction)
                pred_class = LABELS_PHAN_LOAI[pred_class_idx]
                confidence = prediction[pred_class_idx]
                
                # Lưu vào session state
                st.session_state.ripeness_result = {
                    "class": pred_class,
                    "confidence": confidence
                }
                st.session_state.ripeness_probs = {LABELS_PHAN_LOAI[i]: float(prediction[i]) for i in range(len(LABELS_PHAN_LOAI))}
                st.session_state.classification_done = True
        
        # Hiển thị kết quả độ chín
        res = st.session_state.ripeness_result
        probs = st.session_state.ripeness_probs
        
        # Decorate hiển thị kết quả bằng HTML
        color_map = {
            'Chín (Ripe)': '#2E7D32',       # Green
            'Chưa chín (Unripe)': '#F57F17', # Yellow/Orange
            'Chín nẫu (Overripe)': '#D84315' # Dark Orange/Red
        }
        text_color = color_map.get(res["class"], "#000000")
        
        st.markdown(f"""
            <div class='result-card'>
                <h4>Kết quả đánh giá độ chín:</h4>
                <p style='font-size: 1.5rem; font-weight: bold; color: {text_color}; margin: 5px 0;'>
                    🎯 {res["class"]}
                </p>
                <p style='font-size: 1.1rem; color: #555;'>Độ tin cậy: <b>{res["confidence"]:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ phần trăm xác suất từng loại độ chín
        st.write("📊 Phân bố xác suất trạng thái:")
        for label, prob in probs.items():
            st.write(f"{label}: **{prob:.2%}**")
            st.progress(prob)

        # ================= STEP 2: NHẬN DIỆN HƯ HỎNG (TUẦN TỰ) =================
        st.markdown("<div class='step-header'>2️⃣ Quét nhận diện lỗi & bệnh</div>", unsafe_allow_html=True)
        
        # Hiển thị nút bấm để chạy nhận diện lỗi
        if not st.session_state.run_detection:
            st.write("Sau khi đã xem kết quả phân loại độ chín, bạn có thể tiếp tục quét để kiểm tra xem quả có bị lỗi, bệnh hay các vết hư hỏng nào không.")
            if st.button("🔍 Tiến hành quét lỗi & bệnh trên hoa quả", key="btn_detect"):
                st.session_state.run_detection = True
                st.rerun()

        # Thực thi mô hình nhận diện lỗi khi nhấn nút
        if st.session_state.run_detection:
            if not st.session_state.detection_done:
                with st.spinner("Đang phân tích hình ảnh để phát hiện lỗi/bệnh..."):
                    # Tiền xử lý ảnh cho PyTorch
                    transform = T.Compose([T.ToTensor()])
                    img_tensor = transform(img).unsqueeze(0).to(det_device)
                    
                    # Dự đoán Faster R-CNN
                    with torch.no_grad():
                        prediction = model_det(img_tensor)
                    
                    # Trích xuất kết quả dự đoán và đưa về CPU
                    boxes = prediction[0]['boxes'].cpu().numpy()
                    labels = prediction[0]['labels'].cpu().numpy()
                    scores = prediction[0]['scores'].cpu().numpy()
                    
                    # Lưu vào session state
                    st.session_state.detection_preds = {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores
                    }
                    st.session_state.detection_done = True
            
            # Sử dụng kết quả dự đoán đã cache và vẽ hình dựa trên ngưỡng tin cậy (threshold) hiện tại
            preds = st.session_state.detection_preds
            boxes = preds["boxes"]
            labels = preds["labels"]
            scores = preds["scores"]
            
            # Vẽ lên ảnh
            img_to_draw = img.copy()
            draw = ImageDraw.Draw(img_to_draw)
            
            # Cố gắng nạp font tiếng Việt đẹp, nếu không dùng font mặc định
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
                
            defect_count = 0
            for box, label, score in zip(boxes, labels, scores):
                if score >= threshold:
                    class_id = int(label)
                    
                    # Chỉ vẽ nếu không phải là Background (ID 12)
                    if class_id != 12:
                        defect_count += 1
                        box = box.astype(int)
                        class_name = CLASS_NAMES_NHAN_DIEN.get(class_id, f"Lỗi ID {class_id}")
                        
                        # Vẽ Bounding Box (Màu đỏ cho lỗi)
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="#D32F2F", width=4)
                        # Vẽ nhãn chữ nền đỏ chữ trắng để dễ nhìn
                        label_text = f"{class_name}: {score:.2%}"
                        
                        # Lấy kích thước text để vẽ background box cho chữ
                        try:
                            # Pillow 10+
                            text_w = draw.textlength(label_text, font=font)
                            text_h = 18
                        except AttributeError:
                            text_w, text_h = draw.textsize(label_text, font=font)
                            
                        draw.rectangle([(box[0], box[1] - text_h - 4), (box[0] + text_w + 6, box[1])], fill="#D32F2F")
                        draw.text((box[0] + 3, box[1] - text_h - 2), label_text, fill="white", font=font)
            
            # Hiển thị hình ảnh kết quả nhận diện
            st.image(img_to_draw, caption="Ảnh kết quả quét nhận diện lỗi/bệnh", use_column_width=True)
            
            if defect_count > 0:
                st.error(f"⚠️ Phát hiện thấy **{defect_count}** vùng lỗi hoặc bệnh trên bề mặt hoa quả!")
            else:
                st.success("🎉 Tuyệt vời! Không phát hiện thấy lỗi hay bệnh nào trên bề mặt hoa quả với ngưỡng tin cậy này.")
else:
    st.info("Vui lòng tải một hình ảnh hoa quả lên để bắt đầu quy trình đánh giá chất lượng.")
