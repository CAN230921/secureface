#define _GNU_SOURCE
#include <dbus/dbus.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <stdbool.h>
#include <syslog.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FACEAUTH_BUS "io.secureface.FaceAuth"
#define FACEAUTH_PATH "/io/secureface/FaceAuth"
#define FACEAUTH_IFACE "io.secureface.FaceAuth"

static int call_faceauthd(pam_handle_t *pamh, const char *user, const char *reason, int timeout_ms) {
    DBusError err;
    dbus_error_init(&err);

    DBusConnection *conn = dbus_bus_get(DBUS_BUS_SYSTEM, &err);
    if (dbus_error_is_set(&err) || conn == NULL) {
        pam_syslog(pamh, LOG_WARNING, "faceauthd unavailable: %s", err.message ? err.message : "unknown error");
        dbus_error_free(&err);
        return PAM_IGNORE;
    }

    DBusMessage *msg = dbus_message_new_method_call(FACEAUTH_BUS, FACEAUTH_PATH, FACEAUTH_IFACE, "Authenticate");
    if (!msg) {
        pam_syslog(pamh, LOG_ERR, "failed to allocate D-Bus message");
        dbus_connection_unref(conn);
        return PAM_IGNORE;
    }

    dbus_uint32_t timeout = (dbus_uint32_t)timeout_ms;
    if (!dbus_message_append_args(msg,
                                  DBUS_TYPE_STRING, &user,
                                  DBUS_TYPE_STRING, &reason,
                                  DBUS_TYPE_UINT32, &timeout,
                                  DBUS_TYPE_INVALID)) {
        pam_syslog(pamh, LOG_ERR, "failed to append D-Bus args");
        dbus_message_unref(msg);
        dbus_connection_unref(conn);
        return PAM_IGNORE;
    }

    DBusMessage *reply = dbus_connection_send_with_reply_and_block(conn, msg, timeout_ms + 200, &err);
    dbus_message_unref(msg);

    if (dbus_error_is_set(&err) || reply == NULL) {
        pam_syslog(pamh, LOG_WARNING, "faceauthd call failed: %s", err.message ? err.message : "unknown error");
        dbus_error_free(&err);
        dbus_connection_unref(conn);
        return PAM_IGNORE;
    }

    const char *result = NULL;
    dbus_bool_t fallback = true;
    if (!dbus_message_get_args(reply, &err,
                               DBUS_TYPE_STRING, &result,
                               DBUS_TYPE_BOOLEAN, &fallback,
                               DBUS_TYPE_INVALID)) {
        pam_syslog(pamh, LOG_WARNING, "invalid faceauthd response: %s", err.message ? err.message : "unknown");
        dbus_error_free(&err);
        dbus_message_unref(reply);
        dbus_connection_unref(conn);
        return PAM_IGNORE;
    }

    dbus_message_unref(reply);
    dbus_connection_unref(conn);

    if (result && strcmp(result, "PASS") == 0 && fallback == FALSE) {
        pam_syslog(pamh, LOG_INFO, "face authentication succeeded for user %s", user);
        return PAM_SUCCESS;
    }

    // Non-PASS results intentionally do not fail auth; we ignore and let password continue.
    pam_syslog(pamh, LOG_INFO, "face authentication result=%s, fallback=%s", result ? result : "(null)", fallback ? "true" : "false");
    return PAM_IGNORE;
}

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)flags;
    (void)argc;
    (void)argv;

    const char *user = NULL;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || user == NULL) {
        pam_syslog(pamh, LOG_WARNING, "failed to determine PAM user");
        return PAM_IGNORE;
    }

    return call_faceauthd(pamh, user, "pam_auth", 2500);
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    return PAM_SUCCESS;
}
