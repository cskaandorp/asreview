import * as React from "react";
import { useQuery } from "react-query";
import { useParams } from "react-router-dom";

import { Box, Fade, Grid, Stack } from "@mui/material";
import { styled } from "@mui/material/styles";
import Paper from "@mui/material/Paper";

import { PageHeader } from "Components";
import { EndCollaboration, InvitationContents } from ".";

import { TeamAPI } from "api";
import InvitationForm from "./InvitationForm";
import UserListComponent from "./UserListComponent";
import { parseJsonSourceFileConfigFileContent } from "typescript";

const Root = styled("div")(({ theme }) => ({}));

const TeamPage = (props) => {
  const { project_id } = useParams();
  const [selectableUsers, setSelectableUsers] = React.useState([]);
  const [collaborators, setCollaborators] = React.useState([]);
  const [invitedUsers, setInvitedUsers] = React.useState([]);

  const usersQuery = useQuery(["fetchUsers", project_id], TeamAPI.fetchUsers, {
    refetchOnWindowFocus: false,
    onSuccess: (data) => {
      // filter all collaborators and invited people from users
      const associatedUsers = [...data.collaborators, ...data.invitations];
      const allUsers = data.all_users;
      //
      setSelectableUsers((state) =>
        allUsers
          .filter((item) => !associatedUsers.includes(item))
          .sort((a, b) => a.name.toLowerCase() - b.name.toLowerCase())
      );
      setCollaborators((state) => data.collaborators);
      setInvitedUsers((state) => data.invitations);
    },
  });

  const onInvite = (user) => {
    // call api
    // remove user from allUsers
    const index = selectableUsers.findIndex((item) => item.id === user.id);
    setSelectableUsers((state) => [
      ...selectableUsers.slice(0, index),
      ...selectableUsers.slice(index + 1),
    ]);
    // set in Pending invitations
    setInvitedUsers((state) =>
      [...invitedUsers, user].sort(
        (a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1
      )
    );
  };

  // const inviteUser = () => {
  //   if (selectedUser) {
  //     TeamAPI.inviteUser(project_id, selectedUser.id)
  //       .then((data) => {
  //         if (data.success) {
  //           // add this user to the invited users (ofEffect will take care of the rest
  //           // -autocomplete-)
  //           setInvitedUsers((state) => new Set([...state, selectedUser.id]));
  //           // set selected value to null
  //           setSelectedUser(null);
  //         } else {
  //           console.log("Could not invite user -- DB failure");
  //         }
  //       })
  //       .catch((err) => console.log("Could not invite user", err));
  //   }
  // };

  return (
    <Root aria-label="teams page">
      <Fade in>
        <Box>
          <PageHeader header="Team" mobileScreen={props.mobileScreen} />

          <Box className="main-page-body-wrapper">
            <Stack spacing={3} className="main-page-body">
              {!usersQuery.isFetching && props.isOwner && (
                <Box>
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <InvitationForm
                        selectableUsers={selectableUsers}
                        onInvite={onInvite}
                      />
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <UserListComponent
                        header="Collaborators"
                        users={[]}
                      />
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <UserListComponent
                        header="Pending invitations"
                        users={invitedUsers}
                      />
                      {/* <Box className="main-page-body-wrapper">
          {props.isOwner && false && <InvitationContents />}
          {!props.isOwner && false && <EndCollaboration />}
        </Box> */}
                    </Grid>
                  </Grid>
                </Box>
              )}
            </Stack>
          </Box>
        </Box>
      </Fade>
    </Root>
  );
};

export default TeamPage;
