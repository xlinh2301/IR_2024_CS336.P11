/* eslint-disable react/prop-types */
import ImageViewer from "../Image/ImageViewer";

function ImageItem({
  image_path,
  image_similar,
  frame_id,
  video_folder,
  video_id,
  fps,
  textOcr,
  handleShowVideo,
  handleShowSimilarImage,
  textAsr,
}) {
  const arrayToString = (text) => {
    if (Array.isArray(text)) {
      return text.join(" ");
    }
    if (typeof text === "string") {
      return text; // Return as is if it's already a string
    }
    return ""; // Return an empty string for other types or null/undefined
  };
  

  const stringOcr = arrayToString(textOcr);
  return (
    <>
      <ImageViewer src={image_path} width="300px" height="160px" />
      <div className="d-flex justify-content-evenly">
        <div>
          <div>Frame: {frame_id}</div>
          <div>Folder: {video_folder}</div>
          <div>Video: {video_id}</div>
        </div>
        <div className="d-flex flex-column justify-content-around pb-2">
          <button
            className="btn btn-primary"
            onClick={() =>
              handleShowVideo(video_id, frame_id, fps, stringOcr, textAsr)
            }
          >
            Video
          </button>
          <button
            className="btn btn-primary"
            onClick={() => handleShowSimilarImage(image_similar)}
          >
            Similar Image
          </button>
        </div>
      </div>
    </>
  );
}

export default ImageItem;
