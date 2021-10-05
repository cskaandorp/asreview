import React, { useCallback, useMemo } from "react";
import { useDropzone } from "react-dropzone";
import { Button, Typography } from "@mui/material";
import { styled } from "@mui/material/styles";

const PREFIX = "ProjectUploadFile";

const classes = {
  root: `${PREFIX}-root`,
  uploadButton: `${PREFIX}-uploadButton`,
};

const Root = styled("div")(({ theme }) => ({
  [`& .${classes.root}`]: {
    paddingTop: "24px",
  },

  [`& .${classes.uploadButton}`]: {
    marginTop: "26px",
  },
}));

const baseStyle = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: "60px 20px 60px 20px",
  borderWidth: 2,
  borderRadius: 2,
  borderColor: "#eeeeee",
  borderStyle: "dashed",
  outline: "none",
  transition: "border .24s ease-in-out",
};

const activeStyle = {
  borderColor: "#2196f3",
};

const acceptStyle = {
  borderColor: "#00e676",
};

const rejectStyle = {
  borderColor: "#ff1744",
};

const ProjectUploadFile = (props) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length !== 1) {
        console.log("No valid files provided.");
        return;
      }

      // set error to state
      props.setError(null);

      // set the state such that we ca upload the file
      props.setFile(acceptedFiles[0]);
    },
    [props]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
    // acceptedFiles
  } = useDropzone({
    onDrop: onDrop,
    multiple: false,
    accept: ".txt,.tsv,.tab,.csv,.ris,.xlsx",
  });

  const style = useMemo(
    () => ({
      ...baseStyle,
      ...(isDragActive ? activeStyle : {}),
      ...(isDragAccept ? acceptStyle : {}),
      ...(isDragReject ? rejectStyle : {}),
    }),
    [isDragActive, isDragReject, isDragAccept]
  );

  return (
    <Root>
      <div {...getRootProps({ style })}>
        <input {...getInputProps()} />
        <Typography>Drag 'n' drop a file here, or click to a file</Typography>
      </div>

      {props.file !== null && (
        <div>
          <Typography>File '{props.file.path}' selected.</Typography>
          <Button
            variant="contained"
            color="primary"
            disabled={props.upload}
            onClick={() => props.onUploadHandler()}
            className={classes.uploadButton}
          >
            {props.upload ? "Uploading..." : "Upload"}
          </Button>
        </div>
      )}
    </Root>
  );
};

export default ProjectUploadFile;
