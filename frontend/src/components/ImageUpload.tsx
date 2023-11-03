//ImageUpload.tsx
import React, { ChangeEvent } from "react";

type ImageUploadProps = {
  onImageUpload: (file: File) => void;
};

function ImageUpload({ onImageUpload }: ImageUploadProps) {
  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageUpload(file);
    }
  };

  return (
    <div className="input-container">
      <label htmlFor="file" className="custom-file-input">
        Upload Image
      </label>
      <input
        type="file"
        id="file"
        accept="image/*"
        onChange={handleImageChange}
      />
    </div>
  );
}

export default ImageUpload;
