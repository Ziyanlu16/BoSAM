import os
import numpy as np
import torch
import SimpleITK as sitk
import torch.nn.functional as F
import torchio as tio
import cv2
from PIL import Image, ImageDraw, ImageOps
import tempfile

import gradio as gr
from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2


def build_model():
    checkpoint_path = '/mnt/risk2/SAM-Med3D/work_dir/union_train_turbo_best_continue4/sam_model_dice_best.pth'

    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    state_dict = checkpoint['model_state_dict']

    sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to('cuda')
    sam_model.load_state_dict(state_dict)

    return sam_model

sam_model = build_model()

def preprocess_image(image_path):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.expand_dims(image_array, axis=0)

    transform = tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(128, 128, 128)),
    ])

    transformed_subject = transform(image_array)
    image_tensor = torch.tensor(transformed_subject).unsqueeze(0).to('cuda')

    return image_tensor

def overlay_mask_on_image(image_slice, mask_slice, alpha=0.1):
    image_rgb = Image.fromarray(image_slice).convert("RGB")
    image_rgba = image_rgb.convert("RGBA")

    mask_image = Image.new('RGBA', image_rgba.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    mask = (mask_slice > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask, mode='L')
    mask_draw.bitmap((0, 0), mask_pil, fill=(0, 0, 255, int(255 * alpha)))

    combined_image = Image.alpha_composite(image_rgba, mask_image)

    return combined_image


def load_gt3d(image3d_path):
    gt3d_path = image3d_path.replace('_image3D', '_pred4')
    if not os.path.exists(gt3d_path):
        raise FileNotFoundError(f"The file {gt3d_path} does not exist.")
    image = sitk.ReadImage(gt3d_path)
    image_array = sitk.GetArrayFromImage(image)
    return torch.tensor(image_array).unsqueeze(0).unsqueeze(0).float().to('cuda')


def predict(image3D, sam_model=sam_model, points=None, prev_masks=None, num_clicks=5):
    sam_model.eval()

    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    click_points = []
    click_labels = []

    image3D = norm_transform(image3D.squeeze(dim=1))  # (C, W, H, D)
    image3D = image3D.unsqueeze(dim=1)

    image3D = image3D.to('cuda')

    gt3D = load_gt3d('/mnt/risk2/SAM-Med3D/visual/sam_med3d/test_data/COLON/1.3.6.1.4.1.9328.50.4.0068_gt3D.nii.gz')

    if prev_masks is None:
        prev_masks = torch.zeros_like(image3D).to('cuda')
    low_res_masks = F.interpolate(prev_masks.float(), size=(32,32,32))

    with torch.no_grad():
        image_embedding = sam_model.image_encoder(image3D)
    
    for num_click in range(num_clicks):
        with torch.no_grad():
                        
            batch_points, batch_labels = get_next_click3D_torch_2(prev_masks.to('cuda'), gt3D.to('cuda'))

            points_co = torch.cat(batch_points, dim=0).to('cuda')  
            points_la = torch.cat(batch_labels, dim=0).to('cuda')  

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to('cuda'),
            )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding.to('cuda'), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )

            prev_masks = F.interpolate(low_res_masks, size=[128, 128, 128], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)

            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg, medsam_seg_prob

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)
    image = (image * 255).astype(np.uint8)
    return image


def predicts(img_path, sam_model=sam_model):
    img = preprocess_image(img_path)
    prediction, prediction_prob = predict(img, sam_model)
    return prediction, prediction_prob


def save_nifti(prediction, original_image_path):

    original_image = sitk.ReadImage(original_image_path)
    
    output_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    
    output_image.CopyInformation(original_image)
    

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    temp_filename = temp_file.name
    
    sitk.WriteImage(output_image, temp_filename)
    
    return temp_filename


def gr_interface(img_path, sam_model=sam_model, ranges=[(30, 40), (50, 90), (100, 110)]):
    processed_img = preprocess_image(img_path)
    processed_slices = []
    combined_slices = []
    predicted_slices = []

    prediction, prediction_prob = predicts(img_path, sam_model)

    prediction_prob = prediction.astype(np.float32)


    nifti_file_path = save_nifti(prediction_prob, img_path)

    for start_slice, end_slice in ranges:
        for i in range(start_slice, end_slice):
            processed_slice = processed_img[:, :, i].squeeze(0).squeeze(0)
            processed_slices.append(normalize_image(processed_slice.cpu().numpy()))  # 将张量移动到 CPU 上，然后转换为 NumPy 数组，并归一化

            slice_data = np.squeeze(prediction[i, :, :])
            normalized_slice = normalize_image(slice_data)

            # Overlay mask on image
            combined_image = overlay_mask_on_image(processed_slices[-1], normalized_slice)
            combined_slices.append(combined_image)

            predicted_slices.append(normalize_image(slice_data))

    return processed_slices, combined_slices, predicted_slices, nifti_file_path

iface = gr.Interface(
    fn=gr_interface,
    inputs=["file"],
    outputs=["gallery", "gallery", 'gallery', gr.File(label="Download Predicted NIFTI")
],
    title="NIfTI File Slicer",
    description="Upload a NIfTI file, specify start and end slices, and view the predicted slices.",
    examples=["/mnt/risk2/SAM-Med3D/visual/sam_med3d/test_data/COLON/1.3.6.1.4.1.9328.50.4.0013_image3D.nii.gz"]
)

iface.launch(debug=True)

